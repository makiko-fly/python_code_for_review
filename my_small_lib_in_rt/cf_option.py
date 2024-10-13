import sys
import time
from scipy.stats import norm
from numpy import exp, sqrt, log, pi, square
import numpy as np
import pandas as pd
from loguru import logger

# refer to https://github.com/hashABCD/opstrat/blob/main/opstrat/blackscholes.py

def get_hist_sigma(close_ser, method='ewma', halflife=None, decay=None, min_periods=0):
    if pd.isna(close_ser.iloc[-1]):
        return np.nan

    ret_ser = log(close_ser / close_ser.shift(1))
    # print('ret_ser after squaring:\n{}'.format(ret_ser))
    if method == 'simple':
        return ret_ser.std() * sqrt(252)
    elif method == 'ewma':
        if halflife is not None:
            return sqrt(square(ret_ser).ewm(halflife, min_periods=min_periods).mean().iloc[-1]) * sqrt(252)
        elif decay is not None:
            return sqrt(square(ret_ser).ewm(alpha=decay, adjust=True).mean().iloc[-1]) * sqrt(252)

    raise Exception('not supported method: {}'.format(method))



class Option:
    def __init__(self, c_p, cur_price, strike_price, r, t, q=0, r2=None, t_business=None):
        self.c_p = c_p
        self.cur_price = cur_price
        self.strike_price = strike_price
        self.r = r  # r is risk free rate
        if r2 is None:  # r2 is interest charged by broker to short the stock
            self.r2 = r
        else:
            self.r2 = r2
        self.q = q  # q is dividend yield
        self.t = t  # t is calendar days / 365
        if t_business is None:  # t_business is trading days / 365
            self.t_bus = t
        else:
            self.t_bus = t_business

    def get_bs_d1(self, sigma):
        S, K, r, r2, q, t, t_bus = self.cur_price, self.strike_price, self.r, self.r2, self.q, self.t, self.t_bus
        return (log(S/K) + (r2-q) * t + (sigma*sigma/2) * t_bus) / (sigma * sqrt(t_bus))

    def get_bs_price(self, sigma, ret_delta=False):
        N = norm.cdf
        S, K, r, r2, q, t, t_bus = self.cur_price, self.strike_price, self.r, self.r2, self.q, self.t, self.t_bus
        d1 = self.get_bs_d1(sigma)
        d2 = d1 - sigma * sqrt(t_bus)
        if self.c_p == 'c':
            price = S * exp(-q*t) * N(d1) - K * exp(-r*t) * N(d2)
            if ret_delta:
                return price, N(d1)
            else:
                return price
        elif self.c_p == 'p':
            price = K * exp(-r*t) * N(-d2) - S * exp(-q*t) * N(-d1)
            if ret_delta:
                return price, -N(-d1)
            else:
                return price
        else:
            raise Exception('only "c" or "p" is supported')

    def get_bs_iv(self, option_price):
        if np.isnan(option_price) or option_price <= 0:
            return np.nan
        c_p, S, K, r, t = self.c_p, self.cur_price, self.strike_price, self.r, self.t

        min_sigma, max_sigma = 0.001, 5
        floor_price = self.get_bs_price(min_sigma)
        ceiling_price = self.get_bs_price(max_sigma)
        # print('option price: {}, floor price: {}, ceiling price: {}'.format(option_price, floor_price, ceiling_price))
        if option_price < floor_price:
            return min_sigma
        if option_price > ceiling_price:
            return max_sigma

        max_iterations = 50
        precision = 1.0e-5
        sigma_to_try = 0.5
        for i in range(max_iterations):
            # print('trying sigma: {}'.format(_sigma))
            bs_price = self.get_bs_price(sigma_to_try)
            diff = option_price - bs_price
            if abs(diff) < precision:
                break
            vega = S * norm.pdf(self.get_bs_d1(sigma_to_try)) * sqrt(self.t_bus)
            # print('newton, option price: {}, tried sigma: {}, price: {}, diff: {}, S: {}, K: {}, t_bus: {}, vega: {}'.format(
            #     option_price, sigma_to_try, bs_price, diff, self.cur_price, self.strike_price, self.t_bus, vega))
            sigma_to_try += diff / vega

            # added by CF, use brute force version
            if sigma_to_try <= min_sigma or sigma_to_try >= max_sigma:
                break
        if sigma_to_try <= min_sigma or sigma_to_try >= max_sigma:
            sigma_to_try = self.get_bs_iv_brute_force(option_price, min_sigma, max_sigma, step_size=0.001)
        return sigma_to_try

    def get_bs_iv_brute_force(self, option_price, min_sigma, max_sigma, step_size):
        for sigma_to_try in np.arange(min_sigma, max_sigma, step_size):
            theory_price = self.get_bs_price(sigma_to_try)
            diff = option_price - theory_price
            if self.c_p == 'c':  # use first one where diff is negative
                if diff < 0:
                    return sigma_to_try
            elif self.c_p == 'p':
                if diff > 0:
                    return sigma_to_try
            # print('brute force, option_price: {}, sigma_to_try: {}, theory_price: {}, diff: {}'.format(
            #     option_price, sigma_to_try, theory_price, diff))
        return np.nan

    # refer to https://youtu.be/558k7D2alxM
    def get_bs_greeks(self, sigma):
        S, K, r, r2, q, t, t_bus = self.cur_price, self.strike_price, self.r, self.r2, self.q, self.t, self.t_bus
        d1 = self.get_bs_d1(sigma)
        d2 = d1 - sigma * sqrt(t_bus)
        if self.c_p == 'c':
            delta = norm.cdf(d1)
            # theta = (-((S * sigma * exp(-d1 ** 2 / 2)) / (sqrt(8 * pi * t))) - (N_d2 * r * K * exp(-r * t))) / 365
            theta_days = -S * norm.pdf(d1) * sigma / (2 * sqrt(t_bus)) - r * K * exp(-r * t) * norm.cdf(d2)
            theta = theta_days / 365
            rho = K * t * exp(-r * t) * norm.cdf(d2) / 100
        elif self.c_p == 'p':
            delta = - norm.cdf(-d1)
            # theta = (-((S * sigma * exp(-d1 ** 2 / 2)) / (sqrt(8 * pi * t))) + (N_d2 * r * K * exp(-r * t))) / 365
            theta_days = -S * norm.pdf(d1) * sigma / (2 * sqrt(t_bus)) + r * K * exp(-r * t) * norm.cdf(-d2)
            theta = theta_days / 365
            rho = -K * t * exp(-r * t) * norm.cdf(-d2) / 100
        else:
            raise Exception('only "c" or "p" is supported')
        # gamma = (exp(-d1 ** 2 / 2)) / (S * sigma * sqrt(2 * pi * t))
        gamma = norm.pdf(d1) / (S * sigma * sqrt(self.t_bus))
        # vega = (S * sqrt(t) * exp(-d1 ** 2 / 2)) / (sqrt(2 * pi) * 100)
        vega = S * norm.pdf(d1) * sqrt(self.t_bus) / 100
        return {'delta': delta, 'gamma': gamma, 'rho': rho, 'theta': theta, 'vega': vega}

    def get_binomial_price(self, iterations, sigma):
        t_per_step = self.t / float(iterations)
        t_bus_per_step = self.t_bus / float(iterations)
        discount_factor = exp(-(self.r-self.q)*t_per_step)
        u = exp(sigma * sqrt(t_bus_per_step))
        d = 1. / u
        qu = (exp((self.r-self.q) * t_per_step) - d) / (u - d)
        qd = 1 - qu

        # logger.info('u: {}, d: {}, qu: {}, qd: {}'.format(u, d, qu, qd))

        # 正向构建股票价格矩阵
        N, M = iterations, iterations + 1
        mat_len = iterations + 1
        asset_p_tree = np.zeros([mat_len, mat_len])
        for i in range(mat_len):
            for j in range(mat_len):
                asset_p_tree[j, i] = self.cur_price * u**(i-j) * d**j

        # print('after 正向构建股票价格矩阵:\n{}'.format(pd.DataFrame(asset_p_tree)))

        # 最右端终点价格赋值
        option_p_tree = np.zeros([mat_len, mat_len])
        if self.c_p == 'c':
            option_p_tree[:, mat_len-1] = np.maximum(np.zeros(mat_len), (asset_p_tree[:, N] - self.strike_price))
        else:
            option_p_tree[:, mat_len-1] = np.maximum(np.zeros(mat_len), (self.strike_price - asset_p_tree[:, N]))
        # print('after 最右端终点价格赋值:\n{}'.format(pd.DataFrame(option_p_tree)))

        # 逆向回溯计算
        for i in np.arange(M-2, -1, -1):
            for j in range(0, i+1):
                new_value = discount_factor * (qu * option_p_tree[j, i+1] + qd * option_p_tree[j+1, i+1])
                option_p_tree[j, i] = new_value
                # print('after one step, old value 1: {}, old value 2: {}, option price tree: \n{}'.format(
                #     option_p_tree[j, i+1], option_p_tree[j+1, i+1], pd.DataFrame(option_p_tree)))
                # time.sleep(1)
        return option_p_tree[0, 0]


def main():
    # cur_price = 145
    # strike_price = 150
    # days = 30.0
    # sigma = 0.4
    # o = Option(c_p='c', cur_price=cur_price, strike_price=strike_price, r=0.03, t=days/365)
    # print('APPL, cur price: {}, strike_price: {}, volatility: {}, one month call, bs price: {}, binomial price: {}'.
    #       format(cur_price, strike_price, sigma, o.get_bs_price(sigma), o.get_binomial_price(10, sigma)))

    # option_price = 6
    # print('APPL, cur price: {}, strike_price: {}, one month call, option_price: 6, iv: {}'.
    #       format(cur_price, strike_price, o.get_bs_iv(option_price)))

    # cur_price = 1.0530
    # strike_prices = [1.04, 1.05, 1.06, 1.07, 1.08]
    # market_prices = [0.0193, 0.0125, 0.0075, 0.0040, 0.0020]
    # days = 24
    # results = []
    # for k, mkt in zip(strike_prices, market_prices):
    #     o = Option(c_p='c', cur_price=cur_price, strike_price=k, r=0.01, t=days/365)
    #     iv = o.get_bs_iv(mkt)
    #     greeks = o.get_bs_greeks(iv)
    #     result = (k, iv, greeks['delta'], greeks['gamma'], greeks['rho'], greeks['theta'], greeks['vega'])
    #     print(result)
    #     results.append(result)

    # cur_price = 1.06
    # strike_prices = [1.14, 1.15]
    # market_prices = [0.0027, 0.0017]
    # days = 112
    # results = []
    # for k, mkt in zip(strike_prices, market_prices):
    #     o = Option(c_p='c', cur_price=cur_price, strike_price=k, r=0.015, t=days/365)
    #     iv = o.get_bs_iv(mkt)
    #     print(iv)
    #     greeks = o.get_bs_greeks(iv)
    #     # print(o.get_binomial_price(20, iv))
    #     result = (k, iv, greeks['delta'], greeks['gamma'], greeks['rho'], greeks['theta'], greeks['vega'])
    #     print(result)
    #     results.append(result)


    cur_price = 1.0137
    strike_price = 1.05
    days = 91
    o = Option(c_p='c', cur_price=cur_price, strike_price=strike_price, r=0.015, t=days/365)
    iv = o.get_bs_iv(0.0065)
    # print('iv: ', iv)
    # print(o.get_bs_greeks(0.10))
    o.cur_price = 1.04
    print(o.get_bs_price(iv))


    ### Test implied volatility is not valid
    # cur_price = 66.6
    # strike_price = 16.96
    # days = 516
    # o = Option(c_p='c', cur_price=cur_price, strike_price=strike_price, r=0.03, t=days/365, r2=0.06)
    # print(o.get_bs_iv(65))


if __name__ == '__main__':
    main()




