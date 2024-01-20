
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio

import plotly.io as pio
pio.renderers.default = "firefox"


pyo.init_notebook_mode(connected=True)

# get ticker data
#df = pd.read_csv('EURUSD=X.csv')
# Simuler le dataframe df
np.random.seed(43)  # seed pour la reproductibilité
n = 100  # nombre de jours
dates = pd.date_range("20230101", periods=n)
open_prices = np.cumsum(np.random.randn(n)) + 100  # prix d'ouverture
close_prices = open_prices + np.random.randn(n) * 0.5  # prix de clôture
low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.5)  # prix bas
high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.5)  # prix haut
volumes = np.random.randint(1000, 10000, size=n)  # volumes

df = pd.DataFrame({
    'Open': open_prices,
    'Close': close_prices,
    'Low': low_prices,
    'High': high_prices,
    'Volume': volumes
}, index=dates)

# Sélectionner les colonnes requises et les renommer
df = df[['Open', 'Close', 'Low', 'High', 'Volume']]
df.columns = ['Open', 'Close', 'Low', 'High', 'Volume']
# calculate MACD values
df.ta.macd(close='close', fast=12, slow=26, append=True)

# Force lowercase (optional)
df.columns = [x.lower() for x in df.columns]
print(df)

# Construct a 2 x 1 Plotly figure
fig = make_subplots(rows=2, cols=1)

# price Line
fig.append_trace(
    go.Scatter(
        x=df.index,
        y=df['open'],
        line=dict(color='#ff9900', width=1),
        name='open',
        # showlegend=False,
        legendgroup='1',

    ), row=1, col=1
)

# Candlestick chart for pricing
fig.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False

    ), row=1, col=1
)

# Fast Signal (%k)
fig.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macd_12_26_9'],
        line=dict(color='#ff9900', width=2),
        name='macd',
        # showlegend=False,
        legendgroup='2',

    ), row=2, col=1
)

# Slow signal (%d)
fig.append_trace(
    go.Scatter(
        x=df.index,
        y=df['macds_12_26_9'],
        line=dict(color='#000000', width=2),
        # showlegend=False,
        legendgroup='2',
        name='signal'
    ), row=2, col=1
)

# Colorize the histogram values
colors = np.where(df['macdh_12_26_9'] < 0, '#000', '#ff9900')

# Plot the histogram
fig.append_trace(
    go.Bar(
        x=df.index,
        y=df['macdh_12_26_9'],
        name='histogram',
        marker_color=colors,

    ), row=2, col=1
)

# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)

# Update options and show plot
fig.update_layout(layout)
fig.show()

# Construct a 2 x 1 Plotly figure
fig = make_subplots(rows=2, cols=1)

# ... [votre code pour ajouter des tracés à la figure ici]

# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)

# Update options and show plot
fig.update_layout(layout)
fig.show()

# # Définir le facteur d'échelle pour augmenter la résolution (par exemple, 2 pour doubler la résolution)
# scale_factor = 2

# # Enregistrez la figure avec une résolution augmentée
# pio.write_image(fig, 'figure_high_res.jpg', format='jpg', scale=scale_factor)

