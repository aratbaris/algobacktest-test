# -*- coding: utf-8 -*-
"""Simple TA Backtest Significance"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr

def getprices(symbol, start='2020-01-01', end='2022-06-30'):
    data = pdr.DataReader(symbol, 'yahoo', start=start, end=end)
    return data['Adj Close']

def getpositions_sma(prices, window=14):
    sma = prices.rolling(window).mean()
    signals = []
    for day_price, day_sma in zip(prices, sma):
        if day_price < day_sma:
            signals.append(0)
        elif day_price >= day_sma:
            signals.append(1)
        else:
            signals.append(np.nan)
    signals =  pd.Series(signals)
    positions = signals.shift(1)
    return positions


class Test:
    def __init__(self):
        self.prices = []
        self.positions = []

    def detrend_returns(self):  # log returns
        logrets = np.log(self.prices / self.prices.shift(1)).rename('return')
        avg_return = logrets.mean()
        self.detrended_logret = logrets - avg_return

    def run_montecarlo(self, run=1000, seed=101):
        num_days = len(self.detrended_logret)
        mu = self.detrended_logret.mean()
        sigma = self.detrended_logret.std()
        np.random.seed(seed)
        all_simulations = []
        for _ in range(run):
            simulated_returns = np.random.normal(
                loc=mu, scale=sigma, size=num_days)
            all_simulations.append(simulated_returns)
        self.all_simulations = pd.DataFrame(all_simulations)

    def calculate_trade_results(self):
        self.sim_ret = (self.all_simulations @ self.positions.dropna().values) / len(self.positions.dropna()) * 252

    def run(self):
        self.prices
        self.positions
        self.detrend_returns()
        self.detrended_logret = self.detrended_logret[-len(self.positions.dropna()):]
        self.strategy_return = np.mean(self.detrended_logret.values * self.positions.dropna().values)*252
        self.run_montecarlo()
        self.calculate_trade_results()

    def plot_significance(self):
        self.sim_ret.plot(kind='hist',bins=50)
        plt.axvline(self.strategy_return,c='r')
        plt.title('Backtest Significange')
        plt.xlabel('Strategy Avg Annual Returns')
        plt.ylabel('Frequency')
        plt.show()



