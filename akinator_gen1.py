import argparse
import math
import time
import json
import os

# --- Configuração ---
DB_FILE = "pokemon_db.json"

# Formato: (Probabilidade se TIVER a característica, Probabilidade se NÃO TIVER)
LIKELIHOODS = {
    "s":  (0.90, 0.10),  # Sim: 90% chance de ter
    "n":  (0.10, 0.90),  # Não: 10% chance de ter (90% de não ter)
    "i":  (0.50, 0.50),  # Não Sei: Neutro
    "p":  (0.70, 0.30),  # Provavelmente Sim: 70% chance de ter
    "pn": (0.30, 0.70)   # Provavelmente Não: 30% chance de ter
}

# --- Gerenciamento de Dados ---

def carregar_dados():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def salvar_dados(dados):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

# Salva o log de aprendizado para futura calibração de pesos.

def registrar_feedback(pokemon_real, historico_respostas):

    log_file = "learning_log.json"
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pokemon_real": pokemon_real,
        "historico": historico_respostas
    }
    
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except:
            pass
            
    logs.append(entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

# --- Motor Bayesiano (aqui onde a magia acontece)---

class AkinatorBayes:
    def __init__(self, dados):
        self.dados = dados
        self.atributos_utilizados = set()
        self.historico_respostas = {} 
        
        if not dados:
            self.total = 0
            self.probs = []
            return
            
        self.total = len(dados)
        self.probs = [1.0 / self.total] * self.total

    def get_distribution_entropy(self):
        entropy = 0
        for p in self.probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

# Verifica se o pokemon tem a característica desejada.

    def check_feature(self, pokemon, attr, val):

        # 1. Checagem de TIPO
        if attr == "tipo":
            t1 = pokemon.get("tipo", "")
            t2 = pokemon.get("tipo2", "")
            # Se perguntamos "É tipo Fogo?", retorna True se Primary OU Secondary for Fogo
            return (t1 == val) or (t2 == val)
            
        # 2. Checagem de EVOLUÇÃO
        if attr == "evolui":
            evo = pokemon.get("evolui", "")
            if val is True:
                return bool(evo)
            else:
                return not bool(evo)

        # 3. Checagem de COR
        if attr == "cor":
            return pokemon.get("cor") == val

        # 4. Checagem de Booleanos Genéricos (lendario, inicial, bipede, etc)
        # Se a chave não existir no JSON do pokemon, assumimos False por segurança
        return pokemon.get(attr, False) == val

    def atualizar_probabilidades(self, atributo, valor, resposta_codigo):
        """
        Atualiza as probabilidades com base na resposta (s, n, i, p, pn).
        """
        if resposta_codigo not in LIKELIHOODS:
            return # Se inválido, ignora

        p_tem, p_nao_tem = LIKELIHOODS[resposta_codigo]

        for i, pokemon in enumerate(self.dados):
            tem_atributo = self.check_feature(pokemon, atributo, valor)
            
            # Bayes: P(H|E) = P(E|H) * P(H) / P(E)
            # Aqui é calculado apenas o numerador (Likelihood * Prior) e depois normalizado
            
            likelihood = p_tem if tem_atributo else p_nao_tem
            self.probs[i] *= likelihood
            
        # Normalização
        soma_probs = sum(self.probs)
        if soma_probs == 0: 
            self.probs = [1.0/self.total] * self.total
        else:
            self.probs = [p / soma_probs for p in self.probs]

    # --- MÉTODOS ESTÁTICOS DE SIMULAÇÃO (Para Lookahead) ---
    @staticmethod
    def static_calc_entropy(probs):
        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def static_simulate_update(dados, probs, attr, val, resposta_codigo):
        """ Retorna NOVA lista de probs simulada, sem alterar self. """
        new_probs = list(probs)
        
        # Copia da logica de check_feature e likelihood para ser standalone
        
        # Pega likelihoods
        if resposta_codigo not in LIKELIHOODS: return new_probs
        p_tem, p_nao_tem = LIKELIHOODS[resposta_codigo]

        for i, pokemon in enumerate(dados):
            # Check Feature Inline
            tem_atributo = False
            if attr == "tipo":
                tem_atributo = (pokemon.get("tipo") == val) or (pokemon.get("tipo2") == val)
            elif attr == "evolui":
                evo = bool(pokemon.get("evolui"))
                tem_atributo = evo if val else not evo
            elif attr == "cor":
                tem_atributo = (pokemon.get("cor") == val)
            else:
                tem_atributo = (pokemon.get(attr, False) == val)

            likelihood = p_tem if tem_atributo else p_nao_tem
            new_probs[i] *= likelihood

        # Normalização
        soma = sum(new_probs)
        if soma > 0:
            new_probs = [p/soma for p in new_probs]
        
        return new_probs

# Minimax (Minimizar Entropia) com Profundidade e Beam Search.

    def obter_melhor_pergunta_lookahead(self, profundidade=2, beam_width=5):

        if not self.dados: return None
        
        # Passo 1: Gerar TODAS as perguntas possíveis
        candidatas = self._gerar_perguntas_candidatas()
        
        # Passo 2: Calcular Score Base (Ganho de Informação Imediato - Depth 1)
        # Isso serve para fazer o Beam Search (Poda)
        scores_iniciais = []
        for attr, val in candidatas:
            if (attr, val) in self.atributos_utilizados: continue
            
            # Heurística Rápida: Ganho de Informação com "Sim" perfeito
            # Usa a entropia esperada completa do Depth 1 como filtro
            e_esperada = self._calcular_entropia_esperada(self.probs, attr, val)
            scores_iniciais.append( ((attr, val), e_esperada) )
            
        # Ordena pelo MENOR entropia esperada (Melhor ganho)
        scores_iniciais.sort(key=lambda x: x[1])
        
        # Beam Search: Pega apenas as TOP N para aprofundar
        top_candidatas = [x[0] for x in scores_iniciais[:beam_width]]
        
        if profundidade == 1:
            return top_candidatas[0] if top_candidatas else None
            
        # Passo 3: Aprofundar (Depth > 1) nas Top Candidatas
        melhor_pergunta = None
        melhor_score_final = float('inf') # Menor entropia
        
        print(f"--- Iniciando Lookahead Depth {profundidade} em {len(top_candidatas)} candidatos ---")
        
        for perg in top_candidatas:
            attr, val = perg
            
            # Simula resposta SIM (com peso de probabilidade de ocorrer)
            # Probabilidade do usuário dizer SIM: P(Sim) = Somatório(P(H) * P(Sim|H))
            p_user_sim = 0
            for i, p in enumerate(self.probs):
                tem = self.check_feature(self.dados[i], attr, val)
                # Usa 0.9/0.1 do LIKELIHOODS['s']
                mtch = 0.9 if tem else 0.1
                p_user_sim += p * mtch
                
            p_user_nao = 1.0 - p_user_sim
            
            # Ramo SIM
            probs_sim = self.static_simulate_update(self.dados, self.probs, attr, val, 's')
            # Qual seria a melhor entropia alcançável no próximo turno a partir daqui?
            # Chamada Recursiva (Simplify: Apenas pega a entropia resultante se depth acabou,
            # ou faria nova busca. Para Depth 2, basta pegar a entropia do estado resultante ou idealmente
            # o minimo de entropia de uma nova pergunta. 
            # *Correção*: Minimax real busca o MINIMO da entropia apos a MELHOR pergunta do próximo turno.
            min_ent_sim = self.buscar_minima_entropia_futura(probs_sim)
            
            # Ramo NÃO
            probs_nao = self.static_simulate_update(self.dados, self.probs, attr, val, 'n')
            min_ent_nao = self.buscar_minima_entropia_futura(probs_nao)
            
            # Score Ponderado
            score_final = (p_user_sim * min_ent_sim) + (p_user_nao * min_ent_nao)
            
            if score_final < melhor_score_final:
                melhor_score_final = score_final
                melhor_pergunta = perg
                
        return melhor_pergunta

    def _gerar_perguntas_candidatas(self):
        pgs = []
        # 1. Tipos
        tipos = set()
        for p in self.dados:
            if p.get("tipo"): tipos.add(p.get("tipo"))
            if p.get("tipo2"): tipos.add(p.get("tipo2"))
        for t in tipos: pgs.append(("tipo", t))
        
        # 2. Booleanos e Outros
        pgs.append(("evolui", True))
        cores = set(p.get("cor") for p in self.dados if p.get("cor"))
        for c in cores: pgs.append(("cor", c))
        
        bools = ["lendario", "inicial", "bipede", "tem_cauda", "tem_asas", "tem_chifre", "tem_pelo", "flutua", "tem_casco", "evolui_com_pedra"]
        for b in bools: pgs.append((b, True))
        
        return pgs

    def _calcular_entropia_esperada(self, probs_iniciais, attr, val):
        # Calcula entropia imediata (Depth 1)
        p_sim_total = 0
        for i, p in enumerate(probs_iniciais):
            tem = self.check_feature(self.dados[i], attr, val)
            p_sim_total += p * (1.0 if tem else 0.0) # Simplificação binária para performance
            
        p_nao_total = 1.0 - p_sim_total
        
        e_sim = 0
        if p_sim_total > 0:
            probs_sim = self.static_simulate_update(self.dados, probs_iniciais, attr, val, 's')
            e_sim = self.static_calc_entropy(probs_sim)
            
        e_nao = 0
        if p_nao_total > 0:
            probs_nao = self.static_simulate_update(self.dados, probs_iniciais, attr, val, 'n')
            e_nao = self.static_calc_entropy(probs_nao)
            
        return (p_sim_total * e_sim) + (p_nao_total * e_nao)

    def buscar_minima_entropia_futura(self, probs):
        # Dado um estado futuro, qual a Melhor Pergunta que poderiamos fazer lá?
        # Retorna a Entropia Esperada dessa Melhor Pergunta (Min of the Entropy Curve)
        
        # Otimização: Não checar todas, apenas uma amostra ou usar heurística de entropia atual * fator
        # Para Depth 2 "Real", itera as perguntas de novo neste estado probs.
        # Devido a custo, faz um Beam Search pequeno ou apenas calcular a entropia do estado (Depth 1.5)
        
        entropia_minima = float('inf')
        
        # Pega Top 5 perguntas neste novo estado (Beam interno)
        candidatas = self._gerar_perguntas_candidatas()
        
        count = 0
        for attr, val in candidatas:
            if (attr, val) in self.atributos_utilizados: continue
            
            # Calcula Entropia Esperada desta pergunta no futuro
            e = self._calcular_entropia_esperada(probs, attr, val)
            if e < entropia_minima:
                entropia_minima = e
            
            count += 1
            if count > 10: break # Otimização agressiva para Demo
            
        return entropia_minima if entropia_minima != float('inf') else self.static_calc_entropy(probs)


# --- Interface ---

TEMPLATES_PERGUNTAS = {
    "tipo": "O seu Pokémon é do tipo {}?",
    "cor": "O seu Pokémon é predominantemente {}?",
    "evolui": "O seu Pokémon é capaz de evoluir?",
    "lendario": "O seu Pokémon é considerado lendário?",
    "inicial": "O seu Pokémon é um inicial (ou evolução de um)?",
    "bipede": "O seu Pokémon anda sobre duas patas (bípede)?",
    "tem_cauda": "O seu Pokémon possui cauda?",
    "tem_asas": "O seu Pokémon tem asas?",
    "tem_chifre": "O seu Pokémon tem chifres?",
    "tem_pelo": "O seu Pokémon tem pelo?",
    "flutua": "O seu Pokémon flutua ou levita?",
    "tem_casco": "O seu Pokémon tem um casco ou concha?",
    "evolui_com_pedra": "O seu Pokémon evolui usando uma pedra?"
}

def formatar_pergunta(atributo, valor):
    # Perguntas Específicas com Valor (Tipo, Cor)
    if atributo in ["tipo", "cor"]:
        return TEMPLATES_PERGUNTAS[atributo].format(valor)
    
    # Perguntas Booleanas (Apenas checamos a chave)
    if atributo in TEMPLATES_PERGUNTAS:
        return TEMPLATES_PERGUNTAS[atributo]
        
    # Fallback genérico melhorado
    if isinstance(valor, bool):
        return f"O seu Pokémon tem a característica '{atributo}'?"
    return f"O seu Pokémon tem {atributo} igual a {valor}?"

def main():
    parser = argparse.ArgumentParser(description="Pokémon Akinator (Gen 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ativar modo verbose (mostra probabilidades)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("   POKÉMON AKINATOR (GEN 1) - 151 Mons")
    print("="*50)
    print("Responda com:")
    print(" [s] Sim")
    print(" [n] Não")
    print(" [i] Não sei (Ignorar)")
    print(" [p] Provavelmente Sim")
    print(" [pn] Provavelmente Não")
    print("="*50)

    while True:
        dados = carregar_dados()
        if not dados:
            print("Erro: Base de dados vazia! Rode o import_csv.py primeiro.")
            return

        jogo = AkinatorBayes(dados)
        
        input("\nPressione ENTER para começar...")
        
        perguntas_feitas = 0
        vencedor_encontrado = False

        while perguntas_feitas < 20: 
            # Ordenar candidatos por probabilidade
            zipped_candidates = sorted(zip(jogo.dados, jogo.probs), key=lambda x: x[1], reverse=True)
            max_prob = zipped_candidates[0][1]
            
            # Modo Verbose: Top 10
            if args.verbose:
                print(f"\n--- Top 10 Candidatos (Pergunta {perguntas_feitas+1}) ---")
                for i in range(min(10, len(zipped_candidates))):
                    cand, prob = zipped_candidates[i]
                    print(f"{i+1}. {cand['nome']}: {prob*100:.2f}%")
                print("-" * 40)
            
            # Condição de Parada: Incerteza total (< 5%)
            if perguntas_feitas > 5 and max_prob < 0.05:
                print("\nDesisto! Não sei que Pokémon é esse.")
                vencedor_encontrado = False 
                break

            # Checar confiança alta
            if max_prob > 0.90:
                break
            
            melhor_perg = jogo.obter_melhor_pergunta()
            if not melhor_perg:
                break
                
            attr, val = melhor_perg
            jogo.atributos_utilizados.add(melhor_perg)
            
            txt = formatar_pergunta(attr, val)
            
            # Loop de validação de input
            while True:
                resp = input(f"[Q:{perguntas_feitas+1}] {txt} [s/n/i/p/pn]: ").strip().lower()
                if resp in LIKELIHOODS:
                    break
                print("Opção inválida! Use s, n, i, p, ou pn.")
            
            jogo.atualizar_probabilidades(attr, val, resp)
            
            perguntas_feitas += 1
        
        # Resultado Final
        zipped = sorted(zip(jogo.dados, jogo.probs), key=lambda x: x[1], reverse=True)
        vencedor = zipped[0][0]
        prob = zipped[0][1]
        
        if prob < 0.05:
            pass
        else:
            print("\n" + "="*50)
            print(f"🎉 É o **{vencedor['nome'].upper()}**! ({prob*100:.1f}%)")
            print("="*50)
            
            if input("Acertei? (s/n): ").lower() != 's':
                print("...")
                
        if input("\nJogar dnv? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main()
