# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:34:54 2024

@author: AuraGuardian
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import base64
import io
import random
import warnings
import numpy as np
import statsmodels.api as sm
from numpy import datetime64
import yfinance as yf
warnings.filterwarnings("ignore")

# Function to get ticker price (placeholder implementation)
def getTickerPrice(ticker: str, date: datetime64) -> float:
    return random.uniform(1, 100)  # Example implementation

#Use this function to calculate if want to use the real stock prices 

'''
# Function to get ticker price (provided)
def getTickerPrice(ticker: str, date: pd.Timestamp) -> float:
    #print(date.weekday())    
    #If Weekday used the Closing Price
    if date.weekday() < 5:
        ticker_data = yf.Ticker(ticker)
        history = ticker_data.history(start = date, end = date + pd.DateOffset(days= 1))
        #print(not history.empty)
        if not history.empty:
            return history["Close"][0]
        else: 
            return getTickerPrice(ticker, date = (date + pd.DateOffset(days= 1)))
        
    #If Sunday used the Opening Price from Monday 
    elif date.weekday() == 6:       
        ticker_data = yf.Ticker(ticker)
        history = ticker_data.history(start = date + pd.DateOffset(days= 1), end = date + pd.DateOffset(days= 2))    
        if not history.empty:
            return history["Open"][0]
    
    #Friday Closer for Saturday so used Friday Price
    elif date.weekday() == 5: 
        ticker_data = yf.Ticker(ticker)
        history = ticker_data.history(start = pd.to_datetime(date) + pd.DateOffset(days= -1), end = date)
        if not history.empty:
            return history["Close"][0]
'''

# Create benchmark data using normal random variable
# Benchmark (placeholder implementation) in place of S&P500 can be changead can get real SPY data too from YF

def benchmark_Data(seed:int):
    np.random.seed(seed)  # for reproducibility
    dates = pd.date_range(start='2022-01-01', periods=800, freq='D')
    returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))
    benchmark_data = pd.DataFrame({'Date': dates, 'Return': returns})
    #benchmark_data.to_csv('benchmarkData.csv', index=False)


    # Load benchmark data
    #benchmark_data = pd.read_csv("benchmarkData.csv")
    benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
    benchmark_data.set_index('Date', inplace=True)
    benchmark_returns = benchmark_data['Return']
    
    return benchmark_returns


benchmark_returns = benchmark_Data(69)



# Preprocess_data with option to set Size as 1 or Size as the average of the range given we get
# A feature can be added where we can get a randomized output as the Size 
# We can adjust the portfolio based on that by running 

def preprocess_data(df, only_share:bool):
    
    """
    Preprocess the data by parsing dates, renaming columns, calculating prices, and handling trade sizes.
    """
    
    df['Date'] = pd.to_datetime(df['transactionDate'])
    
    df = df.rename(columns={'ticker': 'Symbol', 'type': 'Side'})
    
    df['Price'] = df.apply(lambda row: getTickerPrice(row['Symbol'], row['Date']), axis=1)
    
    df['Side'] = df['Side'].apply(lambda x: 'buy' if 'Purchase' in x else 'sell')

    # Calculate size based on provided amount ranges defaulted to 1 as mentioned inn the email
    df['Size'] = 1
    
    if only_share == 1:
    
        df = df[['Date', 'Symbol', 'Side', 'Size', 'Price']]
        return df
    
    else:
        df['Size'] = 1
        for i in range(len(df)):
            a = df["amount"][i]
            parts = a.replace('$', '').replace(',', '').split('-')
            if len(parts) != 2:
                raise ValueError("String format should be 'number1 - number2'")
            num1 = int(parts[0].strip())
            num2 = int(parts[1].strip())
            average = (num1 + num2) / 2.0
            df['Size'][i] = average / df['Price'][i]
    
        df = df[['Date', 'Symbol', 'Side', 'Size', 'Price']]
    
        return df

#Calculate Trade Metrics


def calculate_trade_metrics(trades, benchmark_returns):

    """
    Calculate Trade Metrics as Listed in the Dashboard
    """
    
    metrics = {}

    trades['PnL'] = 0.0
    trades['Holding Period'] = 0
    trades['Return'] = 0.0
    trades['PnLbygroup'] = 0.0

    
    for index, trade in trades.iterrows():
        if trade['Side'] == 'buy':
            trades.at[index, 'PnL'] = -trade['Price'] * trade['Size']
        elif trade['Side'] == 'sell':
            trades.at[index, 'PnL'] = trade['Price'] * trade['Size']
    
        trades = trades.sort_values(by='Date')
        trades['Cumulative PnL'] = trades['PnL'].cumsum()

    trades = trades.sort_values(by='Date')
    trades['Cumulative PnL'] = trades['PnL'].cumsum()
    
        
    for symbol in trades['Symbol'].unique():
        #Sorting by Time
        symbol_trades = trades[trades['Symbol'] == symbol].sort_values(by='Date')
        
        #Segregating Trades by Groupby here is "Symbol" or tickr
        #Chronological Pairing: By sorting trades by date, 
        #the method ensures that buy and sell trades are paired in the order 
        #they occurred, simulating a real-world trading scenario where earlier 
        #buys are sold first.
        
        buy_trades = symbol_trades[symbol_trades['Side'] == 'buy']
        sell_trades = symbol_trades[symbol_trades['Side'] == 'sell']
        
        #Cpnsidering P
        for i in range(min(len(buy_trades), len(sell_trades))):
         
            buy_trade = buy_trades.iloc[i]
            sell_trade = sell_trades.iloc[i]
            
            pnl = (sell_trade['Price'] - buy_trade['Price']) * buy_trade['Size']
            
            holding_period = abs(sell_trade['Date'] - buy_trade['Date']).days
            
            ret = pnl / buy_trade['Price']

            trades.loc[buy_trade.name, 'PnLbygroup'] = pnl
            trades.loc[buy_trade.name, 'Holding Period'] = holding_period
            trades.loc[buy_trade.name, 'Return'] = ret
       
    #No of Transactions
    metrics['Total Transactions'] = int(len(trades))
    
    #metrics['Total Trades'] = (len(trades[trades['PnL']])!=0)
    
    metrics['Winning Trades'] = int(len(trades[trades['PnLbygroup'] > 0]))
    metrics['Losing Trades'] = int(len(trades[trades['PnLbygroup'] < 0]))
    
    #Metric No 1
    metrics['Metric No. 1: Win Rate'] = (metrics['Winning Trades'] /len(trades[trades['PnLbygroup'] != 0])) * 100
    

    gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_loss = trades[trades['PnL'] < 0]['PnL'].sum()
    
    
    #Metric No 2: Average Profit per Trade
    metrics['Metric No. 2: Average Profit per Trade'] = gross_profit / metrics['Winning Trades'] if metrics['Winning Trades'] > 0 else 0
    
    
    metrics['Average Loss per Trade'] = gross_loss / metrics['Losing Trades'] if metrics['Losing Trades'] > 0 else 0
    
    #Metric No 3: Profit Factor
    metrics['Metric No. 3: Profit Factor'] = gross_profit / abs(gross_loss) if gross_loss < 0 else float('inf')

    
    
    metrics['Total Volume'] = trades['Size'].sum()
    metrics['Average Trade Price'] = trades['Price'].mean()
    metrics['Total Trade Value'] = (trades['Price'] * trades['Size']).sum()
    
    #Metric No 4: PNL
    metrics['Metric No. 4: Profit and Loss (PnL)'] = trades['PnLbygroup'].sum()
    
    #Metric No 5
    metrics['Metric No. 5: Average Holding Period'] = trades['Holding Period'].mean()

    
    average_return = trades['Return'].mean()
    return_std_dev = trades['Return'].std()
    
    
    
    risk_free_rate = 0.02  # Assuming a risk-free rate of 2%
    
    #Metric No 6
    metrics['Metric No 6: Sharpe Ratio'] = (average_return - risk_free_rate) / return_std_dev if return_std_dev != 0 else float('inf')
    
    
    trades = trades.sort_values(by='Date')
    trades['Cumulative PnL'] = trades['PnL'].cumsum()
    
    trades['Cumulative PnLbygroup'] = trades['PnLbygroup'].cumsum()
    peak = trades['Cumulative PnL'].cummax()
    drawdown = (trades['Cumulative PnL'] - peak).min()
    
    #Metric No 7
    metrics['Metric No 7: Maximum Drawdown'] = drawdown


    # Calculate additional metrics that take Benchmark rates as an input in my case)
    
    
    portfolio_returns = trades.set_index('Date')['Return']
    merged_returns = pd.merge(portfolio_returns, benchmark_returns, left_index=True, right_index=True, how='left')
    merged_returns.fillna(0, inplace=True)
    portfolio_returns = merged_returns.iloc[:, 0]
    benchmark_returns = merged_returns.iloc[:, 1]

    # Metric No 8: Sortino Ratio
    
    downside_std_dev = portfolio_returns[portfolio_returns < 0].std()

    metrics['Metric No 8: Sortino Ratio'] = (average_return - risk_free_rate) / downside_std_dev if downside_std_dev != 0 else float('inf')

    # Metric No 9: Using StatsModel OLS to get the Alpha Beta and R^2  
    
    X = sm.add_constant(benchmark_returns)
    model = sm.OLS(portfolio_returns, X).fit()
    
    metrics['R-Squared'] = model.rsquared
    
    metrics['Beta'] = model.params[1]
    
    metrics['Metric No.9: Jensen\'s Alpha'] = model.params[0]
    
    #Metric No.10 Treynor Ratio
    metrics['Metric No.10: Treynor Ratio'] = (average_return - risk_free_rate) / metrics['Beta'] if metrics['Beta'] != 0 else float('inf')
    
    return pd.Series(metrics), trades

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Unsupported file format'])
    except Exception as e:
        return html.Div([f'There was an error processing this file: {e}'])
    
    try:
        preprocessed_df = preprocess_data(df,1)
        trade_metrics, trades = calculate_trade_metrics(preprocessed_df, benchmark_returns)

        pnl_plot = {
            'data': [{
                'x': trades['Date'],
                'y': trades['Cumulative PnL'],
                'type': 'line',
                'name': 'Cumulative PnL'
            }],
            'layout': {
                'title': 'Cumulative PnL Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Cumulative PnL'}
            }
        }

        # Create the metrics table data
        metrics_data = []
        for metric, value in trade_metrics.items():
            metrics_data.append({'Metric': metric, 'Value': f"{value:.2f}"})
        
        # Create the DataTable component
        table = dash_table.DataTable(
            data=metrics_data,
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'}
            ],
            style_table={'overflowX': 'auto'}  # Prevent horizontal scrolling if needed
        )

        return [table, pnl_plot]

    except Exception as e:
        return html.Div([f'Error in data processing: {e}'])


# Initialize the Dash app

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Trade Performance Dashboard Scalar Field"


app.layout = html.Div(className="container mt-4", children=[
    
    html.H1("Trade Performance Dashboard", className="text-center mb-4"),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        className="border border-secondary rounded p-3 mb-4 text-center",
        multiple=False
    ),

    html.Div(id='output-data-upload', className="mb-4"),  # For table
    html.Div(id='pnl-plot-container', className="mb-4"),  # Separate container for the graph
])


@app.callback(
    [Output('output-data-upload', 'children'), Output('pnl-plot-container', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        table, pnl_plot = parse_contents(contents, filename) 
        return table, dcc.Graph(figure=pnl_plot)  
    else:
        return html.Div(['Upload the Test data file to see the metrics.']), dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
