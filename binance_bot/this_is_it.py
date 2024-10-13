import os
import sys
import time
import traceback
import datetime as dt
import math
import cf_util as cu

import pandas as pd
import numpy as np
from binance.client import Client

def get_rsi(ser, halflife=3):
    assert len(ser) > 20, 'price series length should be larger than 20, price_ser:\n{}'.format(ser)
    delta = ser.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # print('got up:\n', up)
    # print('got down:\n', down)
    ema_up = up.ewm(halflife=halflife, adjust=False, min_periods=halflife * 4).mean().iloc[-1]
    ema_down = down.ewm(halflife=halflife, adjust=False, min_periods=halflife * 4).mean().iloc[-1]
    # print('got ema_down: \n{}'.format(ema_down))
    if ema_down == 0:
        return 100
    else:
        rs = ema_up / ema_down
        return 100 - (100 / (1 + rs))
def print_entire_df(df, prompt=''):
    if prompt:
        print(prompt + '\n')
    with pd.option_context('display.max_rows', None):
        print(df)
def get_presicion(sym, tick_size_str):
    tick_size = float(tick_size_str)
    if tick_size >= 1:
        return 2
    precision = 0
    for i in range(8):
        tick_size *= 10
        precision += 1
        if tick_size / 1 > 0.99:
            return precision
    raise Exception('can not find precision for symbol {}, its tick size: {}'.format(sym, tick_size_str))
def hurst_rs(data:list):
    data = list(data)
    """
    使用rs法计算Hurst指数值
    """
    t_i = []
    for r in range(2, (len(data))//2 + 1):
        g = len(data)//r
        x_i_j = [data[i * r : (i+1)*r] for i in range(g)]
        x_i_mean = [sum(x_i)/r for x_i in x_i_j]
        y_i_j = [[x_i_j[i][j] - x_i_mean[i] for j in range(r)] for i in range(g)]
        z_i_j = [[sum(y_i_j[i][ : j + 1])  for j in range(r)] for i in range(g)]
        r_i = [max(z_i_j[i])-min(z_i_j[i]) for i in range(g)]
        s_i = [math.sqrt(sum([ (x_i_j[i][j]-x_i_mean[i])**2 for j in range(r)]) / (r-1)) for i in range(g)]
        rs_i = [r_i[i]/s_i[i] for i in range(g)]
        rs_mean = sum(rs_i)/g
        # t_i.append( math.sqrt(sum([(rs_i[i] - rs_mean)**2 for i in range(g)])/(g-1)) )
        t_i.append(rs_mean)
    return np.polyfit(np.log(np.arange(2, len(data)//2 + 1)), np.log(np.array(t_i)), 1)[0]
def to_timestr(obj=None):
    if obj is None:
        obj = dt.datetime.now()

    if isinstance(obj, int) or isinstance(obj, np.int64):
        if obj / 100_000_000_000 < 1:
           obj = dt.datetime.fromtimestamp(obj)
        else:
            obj = dt.datetime.fromtimestamp(obj/1000)
    return obj.strftime('%Y-%m-%d %H:%M:%S')
def from_timestr(timestr):
    if isinstance(timestr, dt.datetime):
        return timestr
    return dt.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')


class Trader:
    def __init__(self, key, secret, max_pos_num, amount):
        print('== init client...')
        self.client = Client(key, secret)
        self.cur_pos_num = 0
        self.max_pos_num = max_pos_num
        self.amount = amount
        print('== init universe...')
        self.universe = self.get_universe()
        print('== got universe [{}]: {}'.format(len(self.universe), self.universe))
        self.precision_dict = self.get_precision_dict()
        print('== got precision dict[{}]: {}'.format(len(self.precision_dict), self.precision_dict))
        self.account_data = self.client.futures_account()  # for checking if a contract is isolated
        self.server_pos_df = pd.DataFrame()
        self.secinfo_df = pd.read_csv('../data/universe_all.csv')

    def get_universe(self):
        universe_df = pd.read_csv('../data/universe_all.csv')
        universe_df = universe_df.sort_values(by='RetFromLow')
        return list(universe_df['Symbol'])

    def get_precision_dict(self):
        exchange_info = self.client.get_exchange_info()
        sym_info_list = exchange_info['symbols']
        precision_dict = {}
        for sym_info in sym_info_list:
            if sym_info['symbol'] not in self.universe:
                continue
            all_filters = sym_info['filters']
            price_filter = [x for x in all_filters if x['filterType'] == 'PRICE_FILTER'][0]
            precision = get_presicion(sym_info['symbol'], price_filter['tickSize'])
            precision_dict[sym_info['symbol']] = precision
        # some wired changes
        precision_dict['CTKUSDT'] = 3
        precision_dict['DENTUSDT'] = 6
        precision_dict['SRMUSDT'] = 3
        precision_dict['XTZUSDT'] = 3
        precision_dict['BTTUSDT'] = 6
        precision_dict['STMXUSDT'] = 5
        precision_dict['LITUSDT'] = 3
        precision_dict['AKROUSDT'] = 5
        precision_dict['CHRUSDT'] = 4
        precision_dict['TRBUSDT'] = 2
        precision_dict['AXSUSDT'] = 2
        precision_dict['MATICUSDT'] = 4
        precision_dict['EOSUSDT'] = 3
        precision_dict['ALICEUSDT'] = 3
        precision_dict['IOTXUSDT'] = 5
        print(precision_dict['BTCUSDT'])
        sys.exit()
        return precision_dict

    def get_bar_df_dict(self, symbols, interval, bar_num, drop_latest=False):
        bar_df_dict = {}
        for sym in symbols:
            bar_data = self.client.futures_klines(symbol=sym, interval=interval, limit=bar_num)
            bar_df = pd.DataFrame(bar_data, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                                     'CloseTime', 'QuoteAssetVolume', '_1', '_2', '_3', '_4'])
            bar_df = bar_df[['OpenTime', 'Open', 'High', 'Low', 'Close', 'QuoteAssetVolume', 'CloseTime']]
            bar_df['OpenTime'] = bar_df['OpenTime'].map(lambda x: to_timestr(x))
            bar_df['CloseTime'] = bar_df['CloseTime'].map(lambda x: to_timestr(x))
            bar_df['Open'] = bar_df['Open'].astype(float)
            bar_df['High'] = bar_df['High'].astype(float)
            bar_df['Low'] = bar_df['Low'].astype(float)
            bar_df['Close'] = bar_df['Close'].astype(float)
            bar_df['QuoteAssetVolume'] = bar_df['QuoteAssetVolume'].astype(float)
            bar_df['QuoteAssetVolume'] = bar_df['QuoteAssetVolume'].astype(int)
            bar_df = bar_df.rename(columns={'QuoteAssetVolume': 'Amt'})

            bar_df['Ret'] = bar_df['Close'] / bar_df['Close'].shift(1) - 1
            bar_df['Amp'] = (bar_df['High'] - bar_df['Low']) / bar_df['Low']
            bar_df['EwmaAmp'] = bar_df['Amp'].shift(1).ewm(halflife=3, min_periods=12).mean()
            bar_df['BOP'] = (bar_df['Close'] / bar_df['Open'] - 1) / bar_df['EwmaAmp']  # here, let use close - open
            bar_df['RSI'] = bar_df['Open'].rolling(window=24).apply(get_rsi)

            bar_df['Date'] = bar_df['CloseTime'].map(lambda x: x.split(' ')[0])
            bar_df['Symbol'] = sym
            bar_df = cu.move_cols_to_front(bar_df, ['Symbol'])
            if drop_latest:
                bar_df = bar_df.iloc[:-1]  # 5 minutes after hour begins shouldn't be counted as a bar
            # print('bar_df:\n', bar_df)
            # sys.exit(0)
            bar_df_dict[sym] = bar_df
        return bar_df_dict

    def get_server_pos_df(self):
        # get positions from binance server
        self.account_data = self.client.futures_account()
        pos_data_list = []
        for pos_data in self.account_data['positions']:
            if float(pos_data['initialMargin']) > 0 and float(pos_data['entryPrice']) > 0:
                pos_data_list.append(pos_data)
        # print('raw server pos:\n', pos_data_list)
        pos_df = pd.DataFrame(pos_data_list, columns=['symbol', 'entryPrice', 'positionAmt', 'notional', 'leverage',
                                                      'unrealizedProfit', 'isolated'])
        rename_dict = {'symbol': 'Symbol', 'entryPrice': 'EntryPrice', 'positionAmt': 'Quantity', 'notional': 'CurVal',
                       'leverage': 'Leverage', 'unrealizedProfit': 'PNL', 'isolated': 'Isolated'}
        pos_df = pos_df.rename(columns=rename_dict)
        pos_df['EntryPrice'] = pos_df['EntryPrice'].astype(float)
        pos_df['Quantity'] = pos_df['Quantity'].astype(float)
        pos_df['CurVal'] = pos_df['CurVal'].astype(float)
        pos_df['CurPrice'] = pos_df['CurVal'] / pos_df['Quantity']
        pos_df['Ret'] = pos_df['CurPrice'] / pos_df['EntryPrice'] - 1
        # get entry time and stop loss price from order info
        entry_time_dict, sl_time_dict, sl_price_dict = {}, {}, {}
        for sym in pos_df['Symbol']:
            order_list = self.client.futures_get_all_orders(symbol=sym)
            #print('order_list:\n', order_list)
            order_df = pd.DataFrame(order_list, columns=['symbol', 'status', 'price', 'executedQty', 'time', 'stopPrice'])
            order_df = order_df.sort_values(by='time')
            #print('order_df:\n', order_df)
            entry_time_dict[sym] = to_timestr(order_df.loc[order_df['status'] == 'FILLED'].iloc[-1]['time'])
            if len(order_df.loc[order_df['status'] == 'NEW']) > 0:
                sl_time_dict[sym] = to_timestr(order_df.loc[order_df['status'] == 'NEW'].iloc[-1]['time'])
                sl_price_dict[sym] = float(order_df.loc[order_df['status'] == 'NEW'].iloc[-1]['price'])
            else:
                sl_time_dict[sym] = np.nan
                sl_price_dict[sym] = np.nan
        pos_df['EntryTime'] = pos_df['Symbol'].map(lambda x: entry_time_dict[x])
        pos_df['SLTime'] = pos_df['Symbol'].map(lambda x: sl_time_dict[x])
        pos_df['SLPrice'] = pos_df['Symbol'].map(lambda x: sl_price_dict[x])
        pos_df['SLRet'] = pos_df['SLPrice'] / pos_df['EntryPrice'] - 1

        return pos_df[['Symbol', 'EntryTime', 'EntryPrice', 'Quantity', 'CurPrice', 'CurVal', 'PNL',
                       'SLTime', 'SLPrice', 'Ret', 'SLRet', 'Isolated', 'Leverage']]

    def manage_positions(self, server_pos_df, exclude_syms=[]):
        now = dt.datetime.now()
        for _, row in server_pos_df.iterrows():
            sym = row['Symbol']
            quantity = int(row['Quantity'])
            ret = row['CurPrice'] / row['EntryPrice'] - 1

            # whenever return reaches 50%, just call it a decent profit and liquidate
            if ret > 0.50:
                self.client.futures_create_order(symbol=sym, side='SELL', type='MARKET', quantity=quantity)
                # cancel any related orders
                self.cancel_orders_of_symbols([sym])
                self.cur_pos_num -= 1
                continue

            if sym in exclude_syms:
                print('{} in exclude symbols, skip'.format(sym))
                continue

            # first check stop loss
            # if ret < -0.03:
            #     print('  {}, latest return {} is less than -0.03, liquidate...'.format(sym, ret))
            #     self.client.futures_create_order(symbol=sym, side='SELL', type='MARKET', quantity=quantity)
            #     # cancel any related orders
            #     self.cancel_orders_of_symbols([sym])
            #     self.cur_pos_num -= 1
            #     continue

            if ret < 0.05:
                continue
            if pd.isna(row['SLPrice']):
                print('# no previous sl price, set SL price to 1/10 of current ret, row:\n', row.to_frame().T)
                sl_price = row['EntryPrice'] * (1 + ret/10)
                sl_price = round(sl_price, self.precision_dict[sym])
                trigger_price = round(sl_price * 1.002, self.precision_dict[sym])
                self.cancel_orders_of_symbols([sym])
                self.client.futures_create_order(symbol=sym, side='SELL', type='STOP', reduceOnly='true',
                                                 quantity=quantity, workingType='MARK_PRICE', price=sl_price,
                                                 stopPrice=trigger_price)
            # don't move SL after first setup
            # else:
            #     #print('# check if we can move sl price up:\n', row.to_frame().T)
            #     exclude = ['BAKEUSDT', 'IOSTUSDT']
            #     if sym in exclude:  # we are holding these for long term
            #         print('    {} in exclude list, skip'.format(sym))
            #         continue
            #     prev_sl_price = row['SLPrice']
            #     new_sl_price = round(row['EntryPrice'] * (1 + ret/10), self.precision_dict[sym])
            #     trigger_price = round(new_sl_price * 1.002, self.precision_dict[sym])
            #     if new_sl_price > prev_sl_price:
            #         print('  move sl price up from {} to {}:\n{}'.format(prev_sl_price, new_sl_price, row))
            #         self.cancel_orders_of_symbols([sym])
            #         self.client.futures_create_order(symbol=sym, side='SELL', type='STOP', reduceOnly='true', quantity=quantity, workingTye='MARK_PRICE', price=new_sl_price, stopPrice=trigger_price)

    def cancel_orders_of_symbols(self, sym_list):
        open_orders = self.client.futures_get_open_orders()
        for order in open_orders:
            if order['symbol'] in sym_list:
                print('  cancel order for {}'.format(order['symbol']))
                self.client.futures_cancel_order(symbol=order['symbol'], orderId=order['orderId'])

    def cancel_orders_not_in(self, sym_list):
        open_orders = self.client.futures_get_open_orders()
        for order in open_orders:
            if order['symbol'] not in sym_list:
                print('  cancel order for {}'.format(order['symbol']))
                self.client.futures_cancel_order(symbol=order['symbol'], orderId=order['orderId'])

    def is_isolated(self, symbol):
        for pos_data in self.account_data['positions']:
            if pos_data['symbol'] == symbol:
                return pos_data['isolated']
        raise Exception('isolated info not found for {}'.format(symbol))

    # def buy(self, sym, last_price):
    #     if last_price > 300:
    #         print('price larger than 300, skip')
    #         return
    #     if len(self.server_pos_df) > 0 and sym in set(self.server_pos_df['Symbol']):
    #         print('    {} already in our holdings, skip'.format(sym))
    #         return
    #
    #     if self.cur_pos_num >= self.max_pos_num:
    #         print('    reached max positions[{}], skip'.format(self.max_pos_num))
    #         return
    #
    #     unsafe_count = self.server_pos_df['SLPrice'].isna().sum()
    #     if unsafe_count >= 2:
    #         print('    unsafe positions >=1, skip')
    #         return
    #
    #     if not self.is_isolated(sym):
    #         resp = self.client.futures_change_margin_type(symbol=sym, marginType='ISOLATED')
    #         assert resp['code'] == 200, 'change margin type failed, resp: {}'.format(resp)
    #
    #     quantity = int(amount / last_price)  # last price is less than 300, let just use integer quantity
    #     print('== creating order for symbol: {}, quantity: {}, last_price: {}'.format(sym, quantity, last_price))
    #     self.client.futures_create_order(symbol=sym, side='BUY', type='MARKET', quantity=quantity)

    def get_mid_ret_df(self, bar_df_dict):
        # first print hourly median return
        ret_dfs = []
        for sym in self.universe:
            bar_df = bar_df_dict[sym]
            single_ret_df = bar_df.set_index('OpenTime')[['Ret']].copy()
            single_ret_df = single_ret_df.rename(columns={'Ret': sym})
            ret_dfs.append(single_ret_df)
        ret_df = pd.concat(ret_dfs, axis=1)
        corr_df = ret_df.corr()

        ret_df['MidRet'] = ret_df.mean(axis=1)
        mean = ret_df['MidRet'].mean()
        std = ret_df['MidRet'].std()
        ret_df['Zscore'] = (ret_df['MidRet'] - mean) / std
        ret_df = cu.move_cols_to_front(ret_df, ['MidRet', 'Zscore'])
        return ret_df, corr_df

    def exec_strategy(self, bar_df_dict, style='hft_5m'):
        mid_ret_df, corr_df = self.get_mid_ret_df(bar_df_dict)
        print('== market mid ret df:\n', mid_ret_df.iloc[1:])
        # mid_ret_df.to_csv('./mkt_mid_ret.csv')
        corr_ser = corr_df['BTCUSDT']
        # print('== corr_ser:\n', )

        # rank symbols and make the decision
        stat_tp_list = []
        for sym in self.universe:
            bar_df = bar_df_dict[sym]
            last_price = bar_df.iloc[-1]['Close']
            secinfo_row = self.secinfo_df.loc[self.secinfo_df['Symbol'] == sym].iloc[0]
            ret_from_low = secinfo_row['RetFromLow']
            float_coins = secinfo_row['FloatCoins']
            if float_coins == 0 or np.isnan(float_coins):
                mkt_cap = np.nan
            else:
                mkt_cap = float_coins * last_price / 1_000_000_000
            cur_ret, cur_bop, cur_rsi = bar_df.iloc[-1]['Ret'], bar_df.iloc[-1]['BOP'], bar_df.iloc[-1]['RSI']
            today_date = bar_df.iloc[-1]['Date']
            today_bar_df = bar_df.loc[bar_df['Date'] == today_date]
            today_ret = today_bar_df.iloc[-1]['Close'] / today_bar_df.iloc[0]['Open'] - 1
            prev_date = bar_df.loc[bar_df['Date'] < today_date].iloc[-1]['Date']
            prev_bar_df = bar_df.loc[bar_df['Date'] == prev_date]
            prev_ret = prev_bar_df.iloc[-1]['Close'] / prev_bar_df.iloc[0]['Open'] - 1
            corr_with_btc = corr_ser[sym]
            stat_tp_list.append((bar_df.iloc[-1]['OpenTime'], sym, last_price, ret_from_low, mkt_cap, today_ret,
                                 prev_ret, corr_with_btc, cur_ret, cur_bop, cur_rsi))
        stat_df = pd.DataFrame(stat_tp_list, columns=[
            'OpenTime', 'Symbol', 'LastPrice', 'RetFromLow', 'MktCap', 'TodayRet', 'PrevDayRet', 'CorrWithBTC',
            'HourRet', 'BOP', 'RSI'])
        stat_df = stat_df.sort_values(by=['TodayRet'], ascending=False)
        # put BTC and ETH first
        lead_df = stat_df.loc[stat_df['Symbol'].isin(['BTCUSDT', 'ETHUSDT'])].sort_values(by='Symbol')
        print('== BTC and ETC situation: \n', lead_df)
        stat_df = stat_df.loc[~stat_df['Symbol'].isin(['BTCUSDT', 'ETHUSDT'])]

        up_num, down_num = len(stat_df.loc[stat_df['HourRet'] > 0]), len(stat_df.loc[stat_df['HourRet'] < 0])
        up_perc = up_num / len(stat_df)
        print('\n== stat_df, up:{}, down:{}, up_perc:{:.2f}: \n'.format(up_num, down_num, up_perc))
        print_entire_df(stat_df)
        # print(stat_df)
        # stat_df.to_csv('/tmp/stat.csv')
        # top_df = stat_df.loc[stat_df['TodayRet'] > 0.02].iloc[:50]
        # # top_df = stat_df.loc[stat_df['TodayRet'] > 0.02].reset_index(drop=True)
        # if len(top_df) > 0:
        #     print('== top pool [>0.02]:\n', top_df)

        # potential_pool = stat_df.loc[stat_df['CorrWithBTC'] < 0.55]
        # potential_pool = potential_pool.sort_values(by=['TodayRet'], ascending=False)
        # print('== potential_pool:\n', potential_pool)

        # pool_df = stat_df.loc[(stat_df['TodayRet'] < 0.15) & (stat_df['TodayRet'] > 0.02)].iloc[:20]
        # print('== {} picked pool:\n{}'.format(to_timestr(), pool_df))
        # sys.exit(0)

        # for _, row in pool_df.iterrows():
        #     buy_it = False
        #     if row['Ret'] < 0.03 and row['BOP'] > 0.8 and row['RSI'] < 70:
        #         buy_it = True
        #     if buy_it:
        #         print('\n  --->')
        #         print('  buy signal from row: \n{}'.format(row.to_frame().T))
        #         self.buy(row['Symbol'], row['LastPrice'])
        #         time.sleep(1)
        #         # after every buy, update server_pos_df
        #         self.server_pos_df = self.get_server_pos_df()


    def run(self):
        while True:
            self.server_pos_df = self.get_server_pos_df()
            self.cur_pos_num = len(self.server_pos_df)
            if len(self.server_pos_df) > 0:
                print('\n== {} current server positions:\n{}'.format(to_timestr(), self.server_pos_df))
            else:
                print('  {} no server positions, heartbeat...'.format(to_timestr()))

            print('\n== {} manage positions...'.format(to_timestr()))
            self.manage_positions(self.server_pos_df, exclude_syms=['BAKEUSDT'])
            time.sleep(90)

            # self.universe = self.universe[:1]
            bar_df_dict = self.get_bar_df_dict(self.universe, interval='1h', bar_num=48, drop_latest=False)
            self.exec_strategy(bar_df_dict, style='hft_5m')

            time.sleep(180)




if __name__ == '__main__':
    key = 'wA7na53gNxf7Z4wQ93JwAbdgrULNJPECIc5ayAkn3h6VYjwQE66kpJ9bBCNzp0os'
    secret = 'SYmLQTI66RwfFAvfli5XfjvXOoUQBbIvtj9xB43NWSOMqC6EbSPtYHflnMXnOIvx'
    while True:
        try:
            max_pos_num = 10
            amount = 2000
            Trader(key, secret, max_pos_num, amount).run()
        except Exception as e:
            print('================= exception: ')
            traceback.print_exc()
            print('================= sleep some time, then we are back and running')
            time.sleep(60)
