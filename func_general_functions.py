import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
#from pushover import Pushover
from func_general_functions import *
import time
from variables_general import *

#po = Pushover("abkcrum6gvhtukc6y92eqexgrwes1a")
#po.user("uu9g36cgw2kvhawuxxn7fb3fe85hib")

SCRIPT_VERSION = 'V0.4.1'
SCRIPT_MAGIC = 401


def retryable_initialize(max_retries, delay_seconds, terminal_path):
    for attempt in range(1, max_retries + 1):
        if mt5.initialize(terminal_path):
            
            authorized=mt5.login(login = mt_account, password=mt_pass, server=mt_server)
            
            if authorized:
                # display trading account data 'as is'
                print(f'Connected to {mt5.account_info()[0]}')
                
            else:
                print("failed to connect at account #{}, error code: {}".format(mt_account, mt5.last_error()))
                if datetime.now(timezone.utc).hour == 9:
                    time.sleep(1860)
            return True  # If successful, exit the loop and return True
        else:
            print(f"Attempt {attempt} failed to initialize, error code: {mt5.last_error()}")
            #time.sleep(delay_seconds)  # Wait for the specified time before the next attempt

            
    #send_notification('initialisation failed', f'{mt_account} failed to connect')
    raise MaxRetriesExceeded(f"Max retries ({max_retries}) reached. Initialization failed.")

#def send_notification(title, message):
#    msg = po.msg(message)
#    msg.set('title', title)
#    po.send(msg)

def getprofit():
    current_profit = 0
    current_profit = mt5.positions_get()
    if current_profit==None:
        print("No positions on", ", error code={}".format(mt5.last_error()))
    elif len(current_profit)>0:  
        profit = 0 
        for profits in current_profit:
                profit += profits[15]
        return profit

def is_furthest_away_from_zero(num1, num2):
    """
    Determines which of the two numbers is furthest away from 0.
    Returns True if num1 is further away from 0 than num2, and False otherwise.

    :param num1: First number
    :param num2: Second number
    :return: Boolean indicating if num1 is further away from 0 than num2
    """
    return abs(num1) > abs(num2)

def reverse_type(type):
    if type == 0:
        return mt5.ORDER_TYPE_SELL
    elif type == 1:
        return mt5.ORDER_TYPE_BUY

def is_new_hour():
    current_time = datetime.now().strftime("%H:%M")
    return current_time.endswith(":00")

def is_new_day():
    current_time = datetime.now().strftime("%H:%M")
    return current_time.endswith("00:00")

def is_between_3_and_6():
    current_hour = datetime.now().hour
    return 3 <= current_hour < 23

def open_trade(symbol="EURUSD", lot_size=0.1, stop_loss=0, take_profit=0, deviation=5, is_buy=True, timeframe = "1 - 1"):
    
    trade_key = f"{symbol}_{timeframe}"

    if trade_log.get(trade_key, 0) >= maximum_trades:
        print(f"Trade limit reached for {symbol} on {timeframe}. Trade not opened.")
    else:


        # Initialize MetaTrader 5
        try:
            if not retryable_initialize(max_retries=5, delay_seconds=2, terminal_path=terminal_path):
                print("Initialization failed even after retries.")
                #send_notification(f'Script Stopped {mt_account}, {SCRIPT_VERSION}', 'Initialisation failed')
            else:
                print("Initialization successful!")
        except MaxRetriesExceeded as e:
            print(e)
        # Check if the symbol is available in MarketWatch
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"{symbol} not found, cannot call order_check()")
            return
        # Add the symbol if it is not visible
        if not symbol_info.visible:
            print(f"{symbol} is not visible, trying to switch on")
            if not mt5.symbol_select(symbol, True):
                print(f"symbol_select({symbol}) failed, exit")
                return
        # Prepare the trading request
        price = mt5.symbol_info_tick(symbol).ask if is_buy else mt5.symbol_info_tick(symbol).bid
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": (price - price) + stop_loss if is_buy else (price - price) + stop_loss,
            "tp": (price - price) + take_profit if is_buy else (price - price) + take_profit,
            "deviation": deviation,
            "magic": SCRIPT_MAGIC,
            "comment": SCRIPT_VERSION,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # Send the trading request
        result = mt5.order_send(request)
        
        # Check the execution result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"order_send failed, retcode={result.retcode}")
            #send_notification(f'FAILED ENTRY {mt_account}, {SCRIPT_VERSION}', f'{result.comment}')
            result_dict = result._asdict()
            for field, value in result_dict.items():
                print(f"   {field}={value}")
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_field, tradereq_value in traderequest_dict.items():
                        print(f"       traderequest: {tradereq_field}={tradereq_value}")
            print("shutdown() and quit")
            return
        else:
            print(result)
            opened_positions.append([result[2], result[10][10], result[4], result[10][8], result[10][3], result[3]])
            #opened_positions.append(result)
        
        if is_buy:
            print("Opened BUY position with POSITION_TICKET={}".format(result.order))
            #send_notification(f"BUY {mt_account}, {SCRIPT_VERSION}", f"Pair: {symbol} Entry: {price} TP: {(price - price) + take_profit if is_buy else (price - price) + take_profit} SL: {(price - price) + stop_loss if is_buy else (price - price) + stop_loss}")
            trade_log[trade_key] = trade_log.get(trade_key, 0) + 1
            
        else: 
            print("Opened SELL position with POSITION_TICKET={}".format(result.order))
            #send_notification(f"SELL {mt_account}, {SCRIPT_VERSION}", f"Pair: {symbol} Entry: {price} TP: {(price - price) + take_profit if is_buy else (price - price) + take_profit} SL: {(price - price) + stop_loss if is_buy else (price - price) + stop_loss}")
            trade_log[trade_key] = trade_log.get(trade_key, 0) + 1


def close_trade(ticket, symbol, lot, typee, backup=None, message = 'Partials Taken'):
    # Get position details
    print(33)
    print(backup)
    print(33)
    try:
        price = mt5.symbol_info_tick(symbol)[1]
    except TypeError as e:
        # Handle the error
        print("Error:", e)
        price = backup  # or any default value you want to assign
    
    
    # Create close request
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": reverse_type(typee),
        "position": ticket,
        "price": price,
        "deviation": 200,
        "magic": 401,
        "comment": SCRIPT_VERSION,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send close request
    close_result = mt5.order_send(close_request)

    # Check the execution result
    print("Close #{}: {} {} lots at {} ".format(ticket, symbol, lot, price))
    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode={}".format(close_result.retcode))
        #send_notification('Partial Failed', "Order send failed, retcode={}".format(close_result.retcode))
        print("Result:", close_result)
    else:
        pass
        #send_notification(message, "Close #{}: {} {} lots at {} ".format(ticket, symbol, lot, price))

def modify_trade(symbol = None, deviation=20, pos_ticket = None, new_stop = None, new_take = None, is_buy=None):
    global pipp
    # Initialize MetaTrader 5
    
    # Check if the symbol is available in MarketWatch
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found, cannot call order_check()")
        mt5.shutdown()
        return
    # Add the symbol if it is not visible
    if not symbol_info.visible:
        print(f"{symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, exit")
            mt5.shutdown()
            return
    # Prepare the trading request
    price = mt5.symbol_info_tick(symbol).ask if is_buy else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos_ticket,
        "symbol": symbol,
        "sl": new_stop,
        "tp": new_take,
        "deviation": deviation,
        "comment": SCRIPT_VERSION,
        "magic": SCRIPT_MAGIC,
    }
    # Send the trading request
    result = mt5.order_send(request)
    
    # Check the execution result
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send failed, retcode={result.retcode}")
        result_dict = result._asdict()
        for field, value in result_dict.items():
            print(f"   {field}={value}")
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_field, tradereq_value in traderequest_dict.items():
                    print(f"       traderequest: {tradereq_field}={tradereq_value}")
        return
    #else:
        #opened_positions.append([result.order, price, price+sl_increment, price+(sl_increment*2), price+(sl_increment*3)])
    
    if is_buy:
        print("Opened BUY position with POSITION_TICKET={}".format(result.order))
        
        ##send_notification("BUY Position opened", f"Pair: {symbol} Entry: {price} TP: {(price - price) + take_profit if is_buy else (price - price) + take_profit} SL: {(price - price) + stop_loss if is_buy else (price - price) + stop_loss}")
        
    else:
        print("Opened SELL position with POSITION_TICKET={}".format(result.order))
        
        ##send_notification("SELL Position Opened", f"Pair: {symbol} Entry: {price} TP: {(price - price) + take_profit if is_buy else (price - price) + take_profit} SL: {(price - price) + stop_loss if is_buy else (price - price) + stop_loss}")





def find_closest_number(target, numbers, boole):
    closest_number = None
    min_difference = float('inf')  # Initialize with positive infinity to ensure any difference will be smaller
    if boole:
        for number in numbers:
            difference = abs(target - number[1])
            
            if difference < min_difference:
                min_difference = difference
                closest_number = number[1]
        print('closest number found')
        return number[0], number[1]

    elif boole == False:
        for number in numbers:
            difference = abs(target - number[1])
            
            if difference < min_difference:
                min_difference = difference
                closest_number = number[1]

        return number[0], number[1]

def add_values_with_association(permitted_positions, initial_values, associated_value):
    for existing_values, value in permitted_positions:
        if existing_values == initial_values:
            # Values already exist, update the associated value
            value.append(associated_value)
            return

    # Values don't exist, so append to the array with associated value
    permitted_positions.append((initial_values, [associated_value]))

def find_associated_values(permitted_positions, initial_values):
    for existing_values, associated_values in permitted_positions:
        if existing_values == initial_values:
            return len(associated_values)
    # If the initial values are not found, return an empty list or handle it as needed
    return 0

def closest_value_index(array_a, values_view, poi, index, default_value=None):
    sorted_slice = sorted(array_a[values_view:], key=lambda x: abs(x - poi))
    if index >= len(sorted_slice) or index < 0:
        #send_notification('Error', 'Index out of range')
        return default_value
    
    closest_value = sorted_slice[index]
    return array_a[values_view:].index(closest_value) + values_view

def getprofit():
    current_profit = 0
    current_profit = mt5.positions_get()
    if current_profit==None:
        print("No Open Positions", ", error code={}".format(mt5.last_error()))
    elif len(current_profit)>0:  
        profit = 0 
        for profits in current_profit:
            profit += profits[15]
            #if profits[6] == SCRIPT_MAGIC:
            #    profit += profits[15]
        return profit
    
def variable_lot_size(entry=None, stops_loss=None, type=None, risk=None, symbol=None):
    retryable_initialize(3, 5, terminal_path)
    current_account_info = mt5.account_info()
    account_leverage = current_account_info[2]
    current_risk = 0
    if type==0:

        lots = 0
        while current_risk < risk:
            print(f'current risk: {current_risk}')
            lots += 0.2
            print(symbol)
            if symbol == "XAUUSD":
                current_risk = (((entry - stops_loss)*lots)*(account_leverage)) #XAUUSD
            elif symbol == "US30":
                current_risk = (((entry - stops_loss)*lots)*(account_leverage)) #XAUUSD
            else:
                current_risk = (((entry - stops_loss)*lots)*(account_leverage/account_leverage)) * 100000
            
        return round(lots, 1)
    
    elif type==1:
        
        lots = 0
        while current_risk < risk:
            print(f'current risk: {current_risk}')
            lots += 0.2
            print(symbol)
            if symbol == "XAUUSD":
                current_risk = (((stops_loss - entry)*lots)*(account_leverage)) #XAUUSD
            else:
                current_risk = (((stops_loss - entry)*lots)*(account_leverage/account_leverage)) * 100000
            
        return round(lots, 1)

def risk_reward_calc(current_price=None, stop_loss=None, take_profit=None, buy_sell=None):
    if buy_sell == 0:
        if current_price == take_profit:
            return float('inf')  # Infinite reward
        elif current_price == stop_loss:
            return -float('inf')  # Infinite risk
        else:
            return abs((take_profit - current_price) / (current_price - stop_loss)) if take_profit != current_price else float('inf')
    elif buy_sell == 1:
        if current_price == take_profit:
            return float('inf')  # Infinite reward
        elif current_price == stop_loss:
            return -float('inf')  # Infinite risk
        else:
            return abs((current_price - take_profit) / (stop_loss - current_price)) if take_profit != current_price else float('inf')
    else:
        return None  # Invalid input


def get_running_profit(symbol):
    positions_symbol = mt5.positions_get(symbol=symbol)
    
    if positions_symbol is None:
        print("No positions on {}, error code={}".format(symbol, mt5.last_error()))
        return 0
    elif len(positions_symbol) > 0:
        running_profit = 0
        # calculate running profit
        for position in positions_symbol:
            running_profit += position[15]
        
        return running_profit

def updateStatus(myquery=None, newvalues=None, mycol=None):
    try:
        if myquery is None or newvalues is None or mycol is None:
            print("Error: Missing required parameters.")
            return

        # Update the document
        result = mycol.update_one(myquery, newvalues)

        # Check if the update was successful
        if result.modified_count > 0:
            print('Status updated successfully')
        else:
            print('No document found to update')
    except Exception as e:
        print(f'An error occurred: {e}')