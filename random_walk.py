import numpy as np
import matplotlib.pyplot as plt

# 1. Paramètres du marché et de l'actif
S0 = 100.0      # Prix initial de l'action
T = 1.0         # Horizon de temps (1 an)
mu = 0.05       # Rendement annuel espéré (Drift) : 5%
sigma = 0.20    # Volatilité annuelle : 20%

# 2. Paramètres de la simulation de Monte-Carlo
n_steps = 252   # Nombre de jours de trading dans une année
dt = T / n_steps
n_sims = 10     # Nombre de trajectoires à simuler

# 3. Simulation (Mouvement Brownien Géométrique)
# On crée un tableau rempli de zéros pour stocker nos prix (253 lignes, 10 colonnes)
prices = np.zeros((n_steps + 1, n_sims))
prices[0] = S0 # Le premier jour, toutes les trajectoires partent de 100

# Boucle pour calculer le prix jour après jour
for t in range(1, n_steps + 1):
    # Tirage de chocs aléatoires (loi normale standard)
    Z = np.random.standard_normal(n_sims)
    # Formule mathématique d'évolution du prix
    prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# 4. Affichage du graphique
plt.figure(figsize=(10, 6))
plt.plot(prices)
plt.title("Simulation de Monte-Carlo : 10 Trajectoires de Prix (1 an)")
plt.xlabel("Jours de trading")
plt.ylabel("Prix de l'actif (€)")
plt.grid(True)
plt.show()