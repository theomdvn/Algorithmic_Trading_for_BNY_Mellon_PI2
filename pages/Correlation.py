import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('Correlation backtest (random generation)')
#I)

# 1. Définir la matrice de corrélation
n_assets = 5
corr_target = np.full((n_assets, n_assets), 0.98)
np.fill_diagonal(corr_target, 1.0)

# 2. Utiliser la décomposition de Cholesky
chol_decomp = np.linalg.cholesky(corr_target)

# 3. Générer des séries temporelles multivariées
n_days = 3 * 252  # 3 ans de jours ouvrés
mean_returns = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
volatilities = [0.02, 0.015, 0.018, 0.016, 0.017]

random_returns = np.random.randn(n_days, n_assets)
corr_returns = np.dot(random_returns, chol_decomp.T)
corr_returns = corr_returns * volatilities + mean_returns

prices = pd.DataFrame(corr_returns).cumsum()
prices = prices.apply(lambda x: (x + 1) * 100)  

# Nommez les colonnes pour éviter les doublons
prices.columns = ["stock_corr_" + str(i) for i in range(n_assets)]

# Ajouter 3 stocks supplémentaires sans corrélation
n_new_assets = 3
mean_returns_new = [0.0003, 0.0002, 0.0003]
volatilities_new = [0.015, 0.014, 0.013]

for i in range(n_new_assets):
    random_returns_new = np.random.randn(n_days) * volatilities_new[i] + mean_returns_new[i]
    prices_new = pd.Series(random_returns_new).cumsum()
    prices_new = (prices_new + 1) * 100
    prices["stock_uncorr_" + str(i)] = prices_new  # Nommez la nouvelle colonne

#II)

# Visualiser
st.subheader("Stocks with and without correlation")
st.line_chart(prices)

# Calcul de la matrice de corrélation
correlation_matrix = prices.corr()
st.subheader("Correlation Matrix")
st.dataframe(correlation_matrix)

#III)


# Calcul du spread entre les deux premiers stocks
spread = prices["stock_corr_0"] - prices["stock_corr_1"]
st.subheader("Spread between stock_corr_0 and stock_corr_1")
st.line_chart(spread)

# # Affichage de la matrice de corrélation avec une heatmap
st.subheader("Correlation Matrix Heatmap")
st.pyplot(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1).figure)


def trade_on_spread(prices, stock_a, stock_b):
    # Calcul du spread entre les deux stocks choisis
    spread = prices[stock_a] - prices[stock_b]

    # Calcul de la moyenne et de l'écart-type du spread
    mean_spread = spread.mean()
    std_spread = spread.std()

    # Définir le seuil d'impact
    threshold = mean_spread + 2 * std_spread

    # Stratégie de trading basée sur le spread
    positions = []
    open_prices = []
    pnl = []

    for day in range(len(spread)):
        if spread[day] > threshold and not positions:  # Ouvrir un trade
            positions.append(-1)
            open_prices.append(spread[day])
        elif spread[day] <= mean_spread and positions:  # Clôturer un trade
            close_price = spread[day]
            for open_price in open_prices:
                pnl.append(open_price - close_price)
            positions = []
            open_prices = []

    if positions:
        close_price = spread.iloc[-1]
        for open_price in open_prices:
            pnl.append(open_price - close_price)

    total_pnl = sum(pnl)

    # Résumé des opérations de trading
    num_opened_trades = len([p for p in pnl if p > 0])
    num_closed_trades = len(pnl)
    average_pnl_per_trade = total_pnl / num_closed_trades if num_closed_trades != 0 else 0

    st.write("\nRésumé des opérations de trading :")
    st.write(f"- Nombre de trades ouverts : {num_opened_trades}")
    st.write(f"- Nombre de trades clôturés : {num_closed_trades}")
    st.write(f"- PNL moyen par trade : {average_pnl_per_trade:.2f}")
    st.write(f"- PNL total : {total_pnl:.2f}")
    st.write(f"- Trades effectués entre les stocks : {stock_a} et {stock_b}")
    st.write(f"    -> Vendre {stock_a} et Acheter {stock_b} lorsque le spread est au-dessus du seuil.")
    st.write(f"    -> Acheter {stock_a} et Vendre {stock_b} lorsque le spread est en dessous de la moyenne.\n")

    # # Visualisation
    # spread.plot(figsize=(12, 7))
    # plt.axhline(mean_spread, color='green', linestyle='--', label="Moyenne du Spread")
    # plt.axhline(threshold, color='red', linestyle='--', label="Seuil d'Impact")
    # plt.legend()
    # plt.title(f"Spread entre {stock_a} et {stock_b} avec Points de Trading")
    # for day in range(len(spread)):
    #     if spread[day] > threshold:
    #         plt.scatter(day, spread[day], color='red', zorder=5)
    #     elif spread[day] <= mean_spread:
    #         plt.scatter(day, spread[day], color='green', zorder=5)
    # plt.show()
    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 7))
    spread.plot(ax=ax)
    ax.axhline(mean_spread, color='green', linestyle='--', label="Moyenne du Spread")
    ax.axhline(threshold, color='red', linestyle='--', label="Seuil d'Impact")
    ax.legend()
    ax.set_title(f"Spread entre {stock_a} et {stock_b} avec Points de Trading")

    for day in range(len(spread)):
        if spread[day] > threshold:
            ax.scatter(day, spread[day], color='red', zorder=5)
        elif spread[day] <= mean_spread:
            ax.scatter(day, spread[day], color='green', zorder=5)

    # Display the plot using Streamlit
    st.pyplot(fig)

def simulate_gbm(prices, days, dt=1):
    """
    Simule le mouvement brownien géométrique pour la prédiction des prix futurs.
    """
    predictions = {}

    for stock in prices.columns:
        # Calcul du rendement moyen et de la volatilité
        returns = prices[stock].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()

        # Dernier prix connu
        last_price = prices[stock].iloc[-1]

        # Simuler le mouvement brownien géométrique
        simulation = [last_price]
        for day in range(days):
            delta_W = np.random.normal(0, np.sqrt(dt))
            delta_S = mu * simulation[-1] * dt + sigma * simulation[-1] * delta_W
            simulation.append(simulation[-1] + delta_S)

        predictions[stock] = simulation[1:]

    return pd.DataFrame(predictions)


# Appeler la fonction avec les noms des stocks de votre choix
selected_variable1 = st.selectbox("Select Stock 1", prices.columns)
selected_variable2 = st.selectbox("Select Stock 2", prices.columns)

trade_on_spread(prices, selected_variable1, selected_variable2)

# Simuler les prix pour les 365 prochains jours
days_ahead = 365
future_prices = simulate_gbm(prices, days_ahead)

# # Affichage des prédictions
# future_prices.plot(figsize=(15, 7))
# plt.title("Prédictions des Prix des Stocks sur les 365 Prochains Jours")
# plt.xlabel("Jours dans le Futur")
# plt.ylabel("Prix Prédit")
# plt.show()

st.subheader("Predictions of Stock Prices for the Next 365 Days")
st.line_chart(future_prices)

# 1. Calculer les rendements quotidiens
returns = prices.pct_change().dropna()

# 2. Calculer la corrélation des rendements de chaque stock avec le rendement à t-1 des autres stocks
max_corr_value = -1  # valeur initiale
stock_pair = None

for col1 in returns.columns:
    for col2 in returns.columns:
        if col1 != col2:
            correlation = returns[col1].iloc[1:].corr(returns[col2].shift(1).dropna())
            if correlation > max_corr_value:
                max_corr_value = correlation
                stock_pair = (col1, col2)

st.write(f"The pair with the highest correlation at t-1 is: {stock_pair} with a correlation of {max_corr_value:.2f}")

# 3. Stratégie de trading basée sur cette corrélation
positions = []
pnl = []

for i in range(1, len(returns)):
    # Si le stock 1 monte, acheter le stock 2
    if returns[stock_pair[0]].iloc[i] > 0:
        positions.append(returns[stock_pair[1]].iloc[i])

# Calculer le PNL cumulatif
cumulative_pnl = np.cumsum(positions)

# plt.figure(figsize=(12, 6))
# plt.plot(cumulative_pnl)
# plt.title("PNL Cumulatif basé sur la Stratégie de Corrélation à t-1")
# plt.xlabel("Jours")
# plt.ylabel("PNL Cumulatif")
# plt.grid(True)
# plt.show()
st.subheader("Cumulative PnL based on Correlation Strategy at t-1")
st.line_chart(cumulative_pnl)

final_pnl = cumulative_pnl[-1]
st.write(f"Final PnL after the simulation period: {final_pnl:.2f}")