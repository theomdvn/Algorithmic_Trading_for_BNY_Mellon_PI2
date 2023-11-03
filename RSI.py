import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simuler les données
dates = pd.date_range(start="2020-01-02", end="2021-08-31", freq='D')
n = len(dates)
np.random.seed(42)  # creation de la seed
close_prices = 7000 + np.cumsum(np.random.randn(n) * 50)  # Génération d'un mouvement aléatoire autour de 7000$

df_btc = pd.DataFrame(data={'Close': close_prices}, index=dates)

# Print the result
print(df_btc)

# Calculer le changement quotidien du prix de fermeture
change = df_btc["Close"].diff()
change.dropna(inplace=True)

# Créer deux copies de la série de changement
change_up = change.copy()
change_down = change.copy()

# Isoler les gains et les pertes
change_up[change_up<0] = 0
change_down[change_down>0] = 0

# Vérification
change.equals(change_up+change_down)

# Calculer la moyenne mobile des gains et des pertes
avg_up = change_up.rolling(14).mean()
avg_down = change_down.rolling(14).mean().abs()

rsi = 100 * avg_up / (avg_up + avg_down)

# Configurer le style du graphique
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 20)

# Création des graphiques
ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)

# Premier graphique: prix de fermeture
ax1.plot(df_btc['Close'], linewidth=2)
ax1.set_title('Bitcoin Close Price Simulated')

# Deuxième graphique: RSI
ax2.set_title('Relative Strength Index')
ax2.plot(rsi, color='orange', linewidth=1)
ax2.axhline(30, linestyle='--', linewidth=1.5, color='green')
ax2.axhline(70, linestyle='--', linewidth=1.5, color='red')

# Afficher les graphiques
plt.show()