# Quem é esse Pokémon? (Akinator Gen 1)

**Disciplina:** Introdução à Inteligência Artificial  
**Semestre:** 2025.2  
**Professor:** Andre Luis Fonseca Faustino  
**Turma:** T04  

## Integrantes do Grupo
* Jonathas Leonilton de Lima Souza (20200039460)

## Descrição do Projeto
Este projeto consiste em um sistema especialista baseado no jogo **Akinator**, focado em identificar qualquer um dos 151 Pokémons da 1ª Geração (Kanto). O sistema utiliza técnicas de **Inferência Bayesiana (Naive Bayes)** para calcular probabilidades em tempo real e **Teoria da Informação (Entropia de Shannon)** para selecionar estrategicamente as perguntas que maximizam o ganho de informação.

Diferente de uma árvore de decisão estática, o modelo é probabilístico e resiliente a erros do usuário, permitindo respostas graduais ("Provavelmente Sim", "Não Sei") e utilizando **Lookahead (Minimax Depth 2)** para otimizar o caminho de perguntas. Também implementa um módulo de **Aprendizado Dinâmico (Feedback Loop)**, onde o sistema registra erros para recalibrar seus pesos estatísticos futuramente.

**Tecnologias:** Python 3, Flask (Web Framework), HTML5/CSS3.

## Guia de Instalação e Execução

### 1. Clonando o Repositório
```bash
# Clone o repositório
git clone https://github.com/usuario/Projeto3-intro_IA.git

# Entre na pasta do projeto
cd Projeto3-intro_IA
```

### 2. Instalação das Dependências
Certifique-se de ter o **Python 3.x** instalado.

```bash
# Instale o Flask (única dependência externa)
pip install flask
```

### 3. Configuração do Banco de Dados (Opcional)
O projeto já vem com o banco `pokemon_db.json` preenchido.

### 4. Como Executar
Inicie o servidor Flask:
```bash
python app.py
```
Acesse o jogo no navegador em: **http://127.0.0.1:5000**

Também é possível jogar a versão antiga via terminal:
```bash
python akinator_gen1.py --verbose
```

## Estrutura dos Arquivos

* `app.py`: Servidor Web Flask e rotas da API.
* `akinator_gen1.py`: Motor de inferência (Cérebro). Contém a classe `AkinatorBayes`, cálculo de Entropia, Minimax Lookahead e lógica de atualização de probabilidades.
* `pokemon_db.json`: Base de conhecimento com os 151 Pokémons e seus atributos (Tipos, Cor, Evolução, Características Físicas).
* `learning_log.json`: Log de aprendizado gerado pelo Feedback Loop (respostas de usuários reais para calibração).
* `templates/`: Arquivos HTML (`index.html`, `game.html`, `result.html`).
* `static/`: Estilos CSS w imagens (`genio.png`, etc).

## Funcionalidades de IA Implementadas

1.  **Naive Bayes**: Atualização de crenças baseada em evidências (`P(H|E)`). Suporta incerteza com pesos suavizados (Sim=0.9, Provavelmente=0.7).
2.  **Information Gain (Entropia)**: Seleção gulosa da pergunta que mais reduz a incerteza do sistema ($H(X) = - \sum p \log p$).
3.  **Minimax Lookahead (Depth 2)**: Simulação de cenários futuros para evitar "máximos locais" e escolher perguntas que abrem melhores caminhos em 2 turnos.
4.  **Smart Stop & Gap Rule**: Critérios de parada inteligentes baseados em dominância relativa (Líder > 4x Segundo Colocado).
5.  **Shadow Learning**: Coleta de dados supervisionada onde o usuário informa o *Ground Truth* ao final do jogo para refinar o modelo.

## Referências

*   **Dataset Original (Imagens e Tipos):** [Kaggle - Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)
*   **Inspiração Algorítmica:**
    *   [How I built an Akinator style AI using Bayes Theorem](https://medium.com/@iamanuragtiwari101/how-i-built-an-akinator-style-ai-using-bayes-theorem-and-a-little-bit-of-magic-df2273e3f580)
    *   [Building Akinator with Python using Bayes Theorem](https://medium.com/analytics-vidhya/building-akinator-with-python-using-bayes-theorem-216253c98daa)
    *   [StackOverflow: Algorithm behind Akinator](https://stackoverflow.com/questions/13649646/what-kind-of-algorithm-is-behind-the-akinator-game)
    *   [Python Forum: Akinator Logic](https://python-forum.io/thread-1345.html)
    *   [Construct Forum: Game Logic](https://www.construct.net/en/forum/construct-2/how-do-i-18/game-akinator-128403)
*   **Jogo Original:** [Akinator.com](https://pt.akinator.com/)
