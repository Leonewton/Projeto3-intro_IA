
from flask import Flask, render_template, request, jsonify, session
from akinator_gen1 import AkinatorBayes, carregar_dados, LIKELIHOODS, formatar_pergunta, TEMPLATES_PERGUNTAS
import os

app = Flask(__name__)
app.secret_key = "super_secret_pokemon_key"

# --- Dados Globais (Carrega uma vez) ---
DADOS = carregar_dados()

@app.route("/")
def index():
    # Limpa sessão ao iniciar novo jogo
    session.clear()
    return render_template("index.html")

@app.route("/game")
def game():
    # Se não tem jogo iniciado, inicializa
    if "perguntas_feitas" not in session:
        session["perguntas_feitas"] = 0
        session["atributos_utilizados"] = [] # Lista de (attr, val)
        session["probs"] = [1.0 / len(DADOS)] * len(DADOS) # Probabilidades iniciais
        session["historico"] = [] # Para debug/log se quiser

    return render_template("game.html")

@app.route("/api/next_question", methods=["POST"])
def next_question():
    if not DADOS:
        return jsonify({"error": "Banco de dados vazio"}), 500

    # Recupera estado da sessão
    probs = session.get("probs")
    atributos_utilizados_raw = session.get("atributos_utilizados", [])
    # Reconstrói set de tuplas para o motor
    atributos_utilizados = set(tuple(x) for x in atributos_utilizados_raw)
    perguntas_feitas = session.get("perguntas_feitas", 0)

    # Reconstitui objeto Akinator (estado efêmero + probs persistidas)
    jogo = AkinatorBayes(DADOS)
    jogo.probs = probs
    jogo.atributos_utilizados = atributos_utilizados

    # 1. Checa Condições de Parada
    
    # Ordenar candidatos
    zipped = sorted(zip(jogo.dados, jogo.probs), key=lambda x: x[1], reverse=True)
    best_cand = zipped[0][0]
    best_prob = zipped[0][1]

    second_prob = 0
    if len(zipped) > 1:
        second_prob = zipped[1][1]

    # --- CONDIÇÕES DE PARADA INTELIGENTES (SMART STOP) ---
    
    # 1. Certeza Absoluta
    win_absolute = best_prob > 0.80

    # 2. Dominância Relativa (Gap Rule)
    # Se o líder tem > 60% e é 4x mais provável que o segundo colocado.
    win_relative = (best_prob > 0.60 and best_prob > 4 * second_prob)
    
    # 3. Fim de Jogo "Soft" (Após 15 perguntas, fica menos exigente)
    win_late_game = (perguntas_feitas > 15 and best_prob > 0.55 and best_prob > 2 * second_prob)

    if win_absolute or win_relative or win_late_game:
        return jsonify({
            "status": "finished",
            "result": best_cand["nome"],
            "image": best_cand.get("imagem", ""), # Se tiver campo imagem
            "prob": best_prob
        })

    # Derrota (Muito incerto após 10 perguntas)
    if perguntas_feitas > 10 and best_prob < 0.05:
        return jsonify({
            "status": "give_up"
        })
        
    # Limite de perguntas (20)
    if perguntas_feitas >= 20:
        # Chuta o melhor mesmo assim
        return jsonify({
            "status": "finished",
            "result": best_cand["nome"],
            "prob": best_prob,
            "forced": True
        })

    # 2. Obtém Próxima Pergunta (Com Lookahead e Beam Search)
    # Depth=2 significa: Avalia pergunta atual + 1 futuro turno.
    melhor_perg = jogo.obter_melhor_pergunta_lookahead(profundidade=2, beam_width=5)
    
    # Preparar dados de Verbose (Top 10 Candidatos)
    top_candidates = []
    for cand, prob in zipped[:10]:
        top_candidates.append({
            "nome": cand["nome"],
            "prob": round(prob * 100, 2)
        })

    if not melhor_perg:
        # Acabaram as perguntas úteis -> Chuta
        return jsonify({
            "status": "finished",
            "result": best_cand["nome"],
            "prob": best_prob
        })

    attr, val = melhor_perg
    
    # 3. Salva estado temporário (ainda não respondido)
    session["current_question"] = [attr, val]
    
    # 4. Formata Texto
    texto_pergunta = formatar_pergunta(attr, val)

    return jsonify({
        "status": "playing",
        "question": texto_pergunta,
        "question_number": perguntas_feitas + 1,
        "candidates": top_candidates # Debug info
    })

@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.json
    resposta = data.get("answer") # s, n, i, p, pn
    
    current_q = session.get("current_question")
    if not current_q:
        return jsonify({"error": "Nenhuma pergunta ativa"}), 400
        
    attr, val = current_q
    
    # Recupera Akinator
    probs = session.get("probs")
    jogo = AkinatorBayes(DADOS)
    jogo.probs = probs
    
    # Atualiza Probabilidades
    jogo.atualizar_probabilidades(attr, val, resposta)
    
    # Atualiza Sessão
    session["probs"] = jogo.probs
    session["perguntas_feitas"] = session.get("perguntas_feitas", 0) + 1
    
    # Adiciona aos utilizados (set antigo para logica)
    utilizados = session.get("atributos_utilizados", [])
    utilizados.append([attr, val])
    session["atributos_utilizados"] = utilizados
    
    # Salva histórico detalhado para aprendizado
    hist = session.get("historico_anotado", [])
    hist.append({
        "atributo": attr,
        "valor": val,
        "resposta": resposta
    })
    session["historico_anotado"] = hist
    
    # Limpa pergunta atual
    session.pop("current_question", None)
    
    return jsonify({"status": "ok"})

@app.route("/result")
def result():
    cand_nome = request.args.get("pokemon")
    # Busca a lista completa para o dropdown de correção
    lista_nomes = sorted([p["nome"] for p in DADOS])
    return render_template("result.html", pokemon=cand_nome, todos_pokemons=lista_nomes)

@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.json
    pokemon_real = data.get("pokemon_real")
    acertou = data.get("acertou")
    
    historico_detalhado = session.get("historico_anotado", [])
    
    from akinator_gen1 import registrar_feedback
    registrar_feedback(pokemon_real, historico_detalhado)
    
    return jsonify({"status": "logged"})

if __name__ == "__main__":
    app.run(debug=True)
