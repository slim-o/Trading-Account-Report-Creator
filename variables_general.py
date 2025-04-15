from datetime import datetime, timedelta
import pandas as pd

timestamp = 0
#open = 1
#high = 2
#low = 3
#close = 4
imbalance_poi_pos = []
imbalance_poi_neg = []

instance_ID = ''
terminal_id = ''
#myclient = pymongo.MongoClient("mongodb+srv://rpstester1:madeTOtest1@mongochat.p1dacz4.mongodb.net/?retryWrites=true&w=majority")
#mydb = myclient["LUCI_Dashboard"]
#mycol = mydb["instances"]





permitted_positions = []
maximum_trades = 1
trade_log = {}

MAX_PL = 2800
MIN_RR = 0.8

rates = None

number_of_candles = 350
utc_from = datetime.now() + timedelta(hours=200) 
symbol_iteration = 1
class MaxRetriesExceeded(Exception):
    pass

#IC Markets

#V2
'''
mt_account = 51798786
mt_pass = 'ASF77IN3t@F4sK'
mt_server = 'ICMarketsSC-Demo'
terminal_path = 'C:/Program Files/MetaTrader 5 IC Markets (SC)/terminal64.exe'
symbol = [['EURUSD', 5.0], ['USTEC', 10.0], ['US30', 5.0], ['XAUUSD', 1.0]]
max_position_risk = 500
#'''

'''
mt_account = 52617460
mt_pass = 'Sjo^LP9p'
mt_server = 'VantageInternational-Live 6'
terminal_path = 'C:/Program Files/VantageInternational-Live 6/terminal64.exe'
symbol = [['EURUSD', 5.0], ['USTEC', 10.0], ['US30', 5.0], ['XAUUSD', 1.0]]
max_position_risk = 500
#'''

#'''
mt_account = 894979
mt_pass = 'Iq@b8rw&'
mt_server = 'VTMarkets-Demo'
terminal_path = 'C:/Program Files/VT Markets (Pty) MT5 Terminal/terminal64.exe'
symbol = [['EURUSD', 5.0], ['USTEC', 10.0], ['US30', 5.0], ['XAUUSD', 1.0]]
max_position_risk = 500
#'''
'''
mt_account = 1058152246
mt_pass = 'vOGP8foZPi*'
mt_server = 'FTMO-Demo'
terminal_path = 'C:/Program Files/FTMO MetaTrader 5/terminal64.exe'
symbol = [['XAUUSD', 10.0, 0.794966], ['EURUSD', 20.0, 0.800534], ['GBPUSD', 10.0, 0.788514], ['AUDUSD', 2.0, 0.794966], ['AUDCAD', 10.0, 0.794966]]
max_position_risk = 500
#'''
#------------------------------------------------------


opened_positions = []
