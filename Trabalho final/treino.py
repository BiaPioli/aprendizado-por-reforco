import numpy as np
from env import MarketingEnv

env = MarketingEnv("dataset_preparado.csv")

# numero de acoes possiveis
num_actions = len(env.actions)

# Q-table (dicionário)
Q = {}

# transformar estado em chave de dicionário
def get_state_key(state):
    return tuple(state)

def escolher_acao(estado, epsilon=0.1):
    chave = get_state_key(estado)

    if chave not in Q:
        Q[chave] = np.zeros(num_actions)

    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    
    return np.argmax(Q[chave])


# hiperparâmetros
alpha = 0.1     
gamma = 0.9     
episodios = 2000
historico_recompensa = []
historico_acao = []

print("Treinando agente...")

# loop de treinamento
for ep in range(episodios):
    estado = env.reset()
    chave = get_state_key(estado)

    # inicializa linha na Q-table caso nao exista
    if chave not in Q:
        Q[chave] = np.zeros(num_actions)

    # escolher acao
    acao = escolher_acao(estado)

    # executar acao no ambiente
    proximo_estado, recompensa, done, info = env.step(acao)
    chave_prox = get_state_key(proximo_estado)

    if chave_prox not in Q:
        Q[chave_prox] = np.zeros(num_actions)

    # atualização Q-Learning
    Q[chave][acao] += alpha * (
        recompensa + gamma * np.max(Q[chave_prox]) - Q[chave][acao]
    )

    historico_recompensa.append(recompensa)
    historico_acao.append(acao)

    if ep % 200 == 0:
        print(f"Episódio {ep} | Recompensa: {recompensa:.3f}")

print("Treinamento concluído!")
np.save("recompensas.npy", np.array(historico_recompensa))
np.save("acoes.npy", np.array(historico_acao))

print("metricas salvas em: recompensas.npy e acoes.npy")
