import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
st.title('Cointegration historical backtest')

def history(self, period="1mo", interval="1d",
            start=None, end=None, prepost=False, actions=True,
            auto_adjust=True, back_adjust=False,
            proxy=None, rounding=False, tz=None, timeout=None, **kwargs):
    """
    :Parameters:
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
        start: str
            Download start date string (YYYY-MM-DD) or _datetime.
            Default is 1900-01-01
        end: str
            Download end date string (YYYY-MM-DD) or _datetime.
            Default is now
        prepost : bool
            Include Pre and Post market data in results?
            Default is False
        auto_adjust: bool
            Adjust all OHLC automatically? Default is True
        back_adjust: bool
            Back-adjusted data to mimic true historical prices
        proxy: str
            Optional. Proxy server URL scheme. Default is None
        rounding: bool
            Round values to 2 decimal places?
            Optional. Default is False = precision suggested by Yahoo!
        tz: str
            Optional timezone locale for dates.
            (default data is returned as non-localized dates)
        timeout: None or float
            If not None stops waiting for a response after given number of
            seconds. (Can also be a fraction of a second e.g. 0.01)
            Default is None.
        **kwargs: dict
            debug: bool
                Optional. If passed as False, will suppress
                error message printing to console.
    """

    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

usd_pairs = [
    'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X',
    'USDCAD=X', 'NZDUSD=X', 'AUDJPY=X',
    'USDCNY=X', 'USDBRL=X', 'USDINR=X', 'USDRUB=X', 'USDZAR=X',
    'USDTRY=X', 'USDMXN=X', 'USDSEK=X', 'USDNOK=X', 'USDPLN=X',
]

eur_pairs = [
    'EURJPY=X', 'EURCHF=X', 'EURAUD=X',
    'EURCAD=X', 'EURNZD=X', 'EURGBP=X',
    'EURCNY=X', 'EURBRL=X', 'EURINR=X', 'EURRUB=X', 'EURZAR=X',
]


gbp_pairs = [
    'GBPJPY=X', 'GBPEUR=X', 'GBPCHF=X', 'GBPAUD=X',
    'GBPCAD=X', 'GBPNZD=X', 
]
format_colour = lambda x: 'color:red' if x < 0. else 'color:lightgreen'
# Combine the lists
all_pairs = usd_pairs + eur_pairs + gbp_pairs

# Create Ticker objects for each forex pair
forex_tickers = {pair: yf.Ticker(pair) for pair in all_pairs}

start_date = st.date_input("Select start backtest Date",pd.to_datetime("2024-01-09"))
end_date = st.date_input("Select End backtest Date",pd.to_datetime("2024-01-14"))
timeframe = st.selectbox("Select Timeframe", ['5m','1m','2m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'])
# Create a dictionary to store historical data for each pair
historical_data =  {}

# Fetch and store historical data for each pair
for pair, ticker in forex_tickers.items():
    #data = ticker.history(interval='5m', start='2023-12-01', end='2023-12-08')
    data = ticker.history(interval=timeframe, start=start_date, end=end_date)
    #data = ticker.history(interval='5m',period='1w')
    historical_data[pair] = data['Close']
    #print(pair,len(data['Close']))

#historical_data
df_histo = pd.DataFrame(historical_data)
df_histo = df_histo.ffill()
df_histo = df_histo.bfill()

#st.dataframe(df_histo)

def coint_matrix(all_pairs):
# Create a matrix to store p-values for ADF test
    num_pairs = len(all_pairs)
    cointegration_matrix = pd.DataFrame(index=all_pairs, columns=all_pairs)
    cointegration_ratio = pd.DataFrame(index=all_pairs, columns=all_pairs)
    # Perform ADF test and fill in the matrix
    st.subheader('Cointegration matrix calculation : ')
    progress_bar = st.progress(0)
    i=0
    for pair1 in all_pairs:
        for pair2 in all_pairs:
            if pair1 != pair2:  # Avoid testing the same pair against itself
                #print(df_histo[pair1].isna().sum, np.isinf(df_histo[pair1]).sum())
                #print(df_histo[pair2].isna().sum, np.isinf(df_histo[pair2]).sum())
                series1 = df_histo[pair1]
                series2 = df_histo[pair2]
                # Perform ADF test
                crit, p_value, _, _, critical_values, _ = adfuller(series1 - series2)

                if crit < critical_values['1%'] and crit < critical_values['5%'] and crit < critical_values['10%'] and p_value < 0.05 :
                    statio = True
                else:
                    statio =False
                coint = 0
                i+=1
                progress_bar.progress(i/(num_pairs**2-num_pairs))


            # Simulate some work being done
                    #print(pair1,pair2)
                    #print(crit,critical_values)
                    # Store p-value in the matrix

            
                cointegration_matrix.loc[pair1, pair2] = statio
    return cointegration_matrix

#st.dataframe(coint_matrix(all_pairs))

# # Pair information
# pair1 = 'GBPCHF'
# pair2 = 'NZDUSD'

# # Create a figure and axis for the first pair (NZDUSD)
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot NZDUSD on the left axis
# ax1.plot(df_histo['NZDUSD=X'], color='blue', label=pair2)
# ax1.set_xlabel('Date')
# ax1.set_ylabel(pair2, color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Create a second axis for the second pair (GBPCHF) sharing the same x-axis
# ax2 = ax1.twinx()
# ax2.plot(df_histo['GBPCHF=X'], color='green', label=pair1)
# ax2.set_ylabel(pair1, color='green')
# ax2.tick_params(axis='y', labelcolor='green')

# # Add title and legend
# plt.title('Historical Data for NZDUSD and GBPCHF (Dual Y-axis)')
# plt.legend(loc='upper left', bbox_to_anchor=(0.75, 0.95))

# # Display the plot using st.pyplot()
# st.pyplot(fig)

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse

def VAR_pred(pair1,pair2):
    data = pd.DataFrame()
    data[f'{pair1}'] = df_histo[pair1]
    data[f'{pair2}'] = df_histo[pair2]

    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(data) * 0.7)
    train, test = data[0:train_size], data[train_size:]
    #print('train', train)
    #print('test',test)

  # Vérification de la stationnarité des séries temporelles (peut être sauté si vos données sont déjà stationnaires)
    for column in data.columns:
        result = adfuller(data[column])
        st.write(f'ADF Statistic for {column}: {result[0]}, p-value: {result[1]}')

    # Création et ajustement du modèle VAR
    model = VAR(train)
    model_fitted = model.fit()

    # Prévisions sur l'ensemble de test
    lags = model_fitted.k_ar
    st.write('best lags : ', lags)
    forecast = model_fitted.forecast(train.values[-lags:], steps=len(test))

    # Création d'un DataFrame pour les prévisions
    forecast_df = pd.DataFrame(forecast, columns=[f'Forecast_{pair1}', f'Forecast_{pair2}'], index=test.index)

    # Évaluation des performances avec RMSE

    for i, col in enumerate(data.columns):
        rmse_val = rmse(test[col], forecast_df[f'Forecast_{col}'])
        st.write(f'RMSE for {col}: {rmse_val}')

    # Visualisation des résultats
    plt.figure(figsize=(12, 6))
    # Calculate upper and lower bounds based on historical volatility
    historical_volatility = data[f'{pair1}'].std()
    upper_bound = forecast_df[f'Forecast_{pair1}'] + historical_volatility
    lower_bound = forecast_df[f'Forecast_{pair1}'] - historical_volatility

    historical_volatility2 = data[f'{pair2}'].std()
    upper_bound2 = forecast_df[f'Forecast_{pair2}'] + historical_volatility2
    lower_bound2 = forecast_df[f'Forecast_{pair2}'] - historical_volatility2

    # Assume a confidence interval of 99%
    confidence_level = 0.99

    # Calculate confidence intervals
    z_score = 2.576  # for a 95% confidence interval
    forecast_std = forecast_df[f'Forecast_{pair1}'].std()
    margin_of_error = z_score * (forecast_std / np.sqrt(len(test)))

    forecast_std2 = forecast_df[f'Forecast_{pair2}'].std()
    margin_of_error2 = z_score * (forecast_std2 / np.sqrt(len(test)))

    forecast_ci = pd.DataFrame({
        'lower': forecast_df[f'Forecast_{pair1}'] - margin_of_error,
        'upper': forecast_df[f'Forecast_{pair1}'] + margin_of_error
    })

    forecast_ci_upper = pd.DataFrame({
        'lower': upper_bound - margin_of_error, # Lower IC upper
        'upper': upper_bound + margin_of_error
    })

    forecast_ci_lower = pd.DataFrame({
        'lower': lower_bound - margin_of_error,
        'upper': lower_bound + margin_of_error # Upper IC lower
    })

    forecast_ci2 = pd.DataFrame({
        'lower': forecast_df[f'Forecast_{pair2}'] - margin_of_error2,
        'upper': forecast_df[f'Forecast_{pair2}'] + margin_of_error2
    })

    forecast_ci2_upper = pd.DataFrame({
        'lower': upper_bound2 - margin_of_error, # Lower IC upper
        'upper': upper_bound2 + margin_of_error
    })

    forecast_ci2_lower = pd.DataFrame({
        'lower': lower_bound2 - margin_of_error,
        'upper': lower_bound2 + margin_of_error # Upper IC lower
    })


    # Create a figure and axis for the first pair
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot actual and forecast for pair1
    ax1.plot(test.index, test[f'{pair1}'], label=f'Actual {pair1}')
    ax1.plot(test.index, forecast_df[f'Forecast_{pair1}'], label=f'Forecast {pair1}', linestyle='dotted')
    ax1.plot(test.index, upper_bound, label='Upper Bound', linestyle='dashed', color='red')
    ax1.plot(test.index, lower_bound, label='Lower Bound', linestyle='dashed', color='green')
    ax1.fill_between(test.index, forecast_ci['lower'], forecast_ci['upper'], color='gray', alpha=0.2, label='99% Confidence Interval')
    ax1.fill_between(test.index, forecast_ci_upper['lower'], forecast_ci_upper['upper'], color='red', alpha=0.2, label='99% Confidence Interval')
    ax1.fill_between(test.index, forecast_ci_lower['lower'], forecast_ci_lower['upper'], color='green', alpha=0.2, label='99% Confidence Interval')

    # Add legend and title
    ax1.legend()
    ax1.set_title(f'VAR Model - Forecast vs Actual for {pair1}')

    # Display the plot for pair1 using st.pyplot()
    st.pyplot(fig)

    # Create a figure and axis for the second pair
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Plot actual and forecast for pair2
    ax2.plot(test.index, test[f'{pair2}'], label=f'Actual {pair2}')
    ax2.plot(test.index, forecast_df[f'Forecast_{pair2}'], label=f'Forecast {pair2}', linestyle='dotted')
    ax2.plot(test.index, upper_bound2, label='Upper Bound', linestyle='dashed', color='red')
    ax2.plot(test.index, lower_bound2, label='Lower Bound', linestyle='dashed', color='green')
    ax2.fill_between(test.index, forecast_ci2['lower'], forecast_ci2['upper'], color='gray', alpha=0.2, label='99% Confidence Interval')
    ax2.fill_between(test.index, forecast_ci2_upper['lower'], forecast_ci2_upper['upper'], color='red', alpha=0.2, label='99% Confidence Interval')
    ax2.fill_between(test.index, forecast_ci2_lower['lower'], forecast_ci2_lower['upper'], color='green', alpha=0.2, label='99% Confidence Interval')

    # Add legend and title
    ax2.legend()
    ax2.set_title(f'VAR Model - Forecast vs Actual for {pair2}')

    # Display the plot for pair2 using st.pyplot()
    st.pyplot(fig2)
    return forecast_df,forecast_ci,forecast_ci_upper['lower'],forecast_ci_lower['upper'],forecast_ci2,forecast_ci2_upper['lower'],forecast_ci2_lower['upper']


def backtest(pair1,pair2,df):
    df_histo = df
    forecast ,pred_IC ,UB_lower ,LB_upper , pred_IC_2, UB_lower_2, LB_upper_2 = VAR_pred(pair1,pair2)
    achat = 0
    vente = 0
    ptf = 100
    L_ptf = []
    entry = []
    entry_price = []
    close = []
    close_price = []
    type_trade = []
    pair_traded = []
    for i in range(1,3):
        if i == 1:
            in_trade_long = False
            in_trade_short = False
            pair = pair1
            IC = pred_IC
            UB = UB_lower
            LB = LB_upper
        else:
            in_trade_long = False
            in_trade_short = False
            pair = pair2
            IC = pred_IC_2
            UB = UB_lower_2
            LB = LB_upper_2

        for i in range(len(forecast)):
            if in_trade_long == True or in_trade_short == True:
                if df_histo[pair].tail(len(forecast)).iloc[i] > IC['lower'].iloc[i] and in_trade_long == True:  #Close long
                    in_trade_long = False
                    vente = df_histo[pair].tail(len(forecast)).iloc[i]
                    ptf += ptf*((vente-achat)/abs(achat))
                    L_ptf.append(ptf)
                    close.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i],
                    close_price.append(df_histo[pair].tail(len(forecast)).iloc[i])

                elif df_histo[pair].tail(len(forecast)).iloc[i] < IC['upper'].iloc[i] and in_trade_short == True:  #Close short
                    in_trade_short = False
                    achat = df_histo[pair].tail(len(forecast)).iloc[i]
                    ptf += ptf*((vente-achat)/abs(achat))
                    L_ptf.append(ptf)
                    close.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i],
                    close_price.append(df_histo[pair].tail(len(forecast)).iloc[i])
            else:
                if df_histo[pair].tail(len(forecast)).iloc[i] < LB.iloc[i]: #Long gbp
                    in_trade_long = True
                    achat = df_histo[pair].tail(len(forecast)).iloc[i]
                    entry.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i],
                    entry_price.append(df_histo[pair].tail(len(forecast)).iloc[i])
                    type_trade.append('Long')
                    pair_traded.append(pair)
                elif df_histo[pair].tail(len(forecast)).iloc[i] > UB.iloc[i]: #Short gbp
                    in_trade_short = True
                    vente = df_histo[pair].tail(len(forecast)).iloc[i]
                    entry.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i]
                    entry_price.append(df_histo[pair].tail(len(forecast)).iloc[i])
                    type_trade.append('Short')
                    pair_traded.append(pair)
        # if in_trade_long == True or in_trade_short == True:
        #     close.append('NaN')
        #     L_ptf.append('NaN')
        if in_trade_long == True or in_trade_short == True:
            if in_trade_long == True:  #Close long, df_histo[pair].tail(len(forecast)).iloc[i] > IC['lower'].iloc[i] and 
                    in_trade_long = False
                    #vente = df_histo[pair].tail(len(forecast)).iloc[i]
                    vente = df_histo[pair].iloc[-1]
                    ptf += ptf*((vente-achat)/abs(achat))
                    L_ptf.append(ptf)
                    close.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i],
                    close_price.append(df_histo[pair].iloc[-1])

            elif in_trade_short == True:  #Close short, df_histo[pair].tail(len(forecast)).iloc[i] < IC['upper'].iloc[i] and 
                    in_trade_short = False
                    #achat = df_histo[pair].tail(len(forecast)).iloc[i]
                    achat = df_histo[pair].iloc[-1]
                    ptf += ptf*((vente-achat)/abs(achat))
                    L_ptf.append(ptf)
                    close.append((df_histo[pair].tail(len(forecast)).index[i]))#df_histo[pair].tail(len(forecast)).iloc[i],
                    close_price.append(df_histo[pair].iloc[-1])
    
    df_res = pd.DataFrame()
    df_res['Pair'] = pd.Series(pair_traded)
    df_res['Trade'] = pd.Series(type_trade)
    df_res['Entry'] = pd.Series((entry))
    df_res['Entry price'] = pd.Series((entry_price))
    df_res['Close'] = pd.Series((close))
    df_res['Close price'] = pd.Series((close_price))
    #df_res['Close'] = pd.to_datetime(df_res['Close'])
    df_res['PnL'] = pd.Series(L_ptf)


    return df_res,ptf

def show_error(pair1,pair2):
    pred,_ ,_ , _, _, _, _ = VAR_pred(pair1,pair2)

    error_pair1 = pred[f'Forecast_{pair1}'] - df_histo[f'{pair1}'].tail(len(pred))
    error_pair2 = pred[f'Forecast_{pair2}'] - df_histo[f'{pair2}'].tail(len(pred))
    error_df = pd.DataFrame({f'Error {pair1}': error_pair1.values, f'Error {pair2}': error_pair2.values}, index=pred.index)
    # Plot the errors
    st.line_chart(error_df)


coint = coint_matrix(all_pairs)
true_indices = coint[coint].stack().index.tolist()
# Extract row indices and column names
if len(true_indices) > 0:
    row_indices, column_names = zip(*true_indices)
    # Display the results
    result_df = pd.DataFrame({
        'Row Index': row_indices,
        'Column Name': column_names
    })
    # Display the result DataFrame
    st.dataframe(result_df)

    for p1,p2 in zip(row_indices,column_names):
        df_res,pt = backtest(p1,p2,df_histo)
        st.table(df_res)
        #st.write(f'Final PnL {pt}')
        #show_error(p1,p2)
else:
    st.write('Try another time frame, no cointegration detected')
# #gbpchf_nzdusd,_ ,_ , _, _, _, _ = VAR_pred('GBPCHF=X','NZDUSD=X')
# df_res,pt = backtest('GBPCHF=X','NZDUSD=X',df_histo)
# st.table(df_res)
# #gbpcad_nzdusd,_ ,_ , _, _, _, _= VAR_pred('GBPCAD=X','NZDUSD=X')
# df_res,pt = backtest('GBPCAD=X','NZDUSD=X',df_histo)
# st.table(df_res)
# #eurcad_gbpchf,_ ,_ , _, _, _, _= VAR_pred('EURCAD=X','GBPCHF=X')
# df_res,pt = backtest('EURCAD=X','GBPCHF=X',df_histo)
# st.table(df_res)
# #eurcad_eurchf,_ ,_ , _, _, _, _ = VAR_pred('EURCAD=X','EURCHF=X')
# df_res,pt = backtest('EURCAD=X','EURCHF=X',df_histo)
# st.table(df_res)
# #gbpchf_gbpcad,_ ,_ , _, _, _, _= VAR_pred('GBPCHF=X','GBPCAD=X')
# df_res,pt = backtest('GBPCHF=X','GBPCAD=X',df_histo)
# st.table(df_res)




# show_error('GBPCHF=X','NZDUSD=X')
# show_error('GBPCAD=X','NZDUSD=X')
# show_error('EURCAD=X','GBPCHF=X')
# show_error('EURCAD=X','EURCHF=X')
# show_error('GBPCHF=X','GBPCAD=X')