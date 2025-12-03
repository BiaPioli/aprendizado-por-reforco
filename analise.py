import numpy as np
import matplotlib.pyplot as plt

recompensas = np.load("recompensas.npy")
acoes = np.load("acoes.npy")

# grafico 1: recompensa por episódio
plt.figure(figsize=(10,5))
plt.plot(recompensas)
plt.title("Recompensa por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()

# grafico 2: media móvel 
janela = 100 
media_movel = np.convolve(recompensas, np.ones(janela)/janela, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(media_movel, color='orange')
plt.title("Recompensa Média Móvel (Janela = 100 episódios)")
plt.xlabel("Episódio")
plt.ylabel("Recompensa média")
plt.grid(True)
plt.show()

# grafico 3: distribuição das acoes escolhidas
plt.figure(figsize=(7,4))
plt.hist(acoes, bins=4, rwidth=0.7)
plt.title("Distribuição das Ações Escolhidas")
plt.xlabel("Ação")
plt.ylabel("Frequência")
plt.xticks([0, 1, 2, 3], ["Nenhuma", "Email", "Anúncio", "Redes Sociais"])
plt.grid(axis='y')
plt.show()

# 4. recompensa média por ação
recompensa_por_acao = {}
for a in [0, 1, 2, 3]:
    indices = np.where(acoes == a)[0]
    if len(indices) > 0:
        recompensa_por_acao[a] = recompensas[indices].mean()

print("Recompensa média por ação:")
for a, valor in recompensa_por_acao.items():
    nome = ["Nenhuma", "Email", "Anúncio", "Redes Sociais"][a]
    print(f"Ação {a} ({nome}): {valor:.4f}")
