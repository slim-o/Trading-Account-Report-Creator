from func_general_functions import *
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import math
from dash import Dash, html, dash_table, dcc, callback, Output, Input, exceptions, callback_context
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import State

pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 20000)     # max table width to display
pd.set_option('display.max_rows', 10)

# Global variables to store cached data
cached_data = None
last_refresh = None
account_info = None
starting_balance = 0
mt_connected = False
mt_server = None
mt_account = None
mt_pass = None
terminal_path = None

extra_text = ''
gross_profit = 0
gross_loss = 0
long_wins = 0
short_wins = 0
long_loss = 0
short_loss = 0
win_rate = 0
largest_profit = 0
largest_loss = 0
profit_factor = 0
recovery_factor = 0
sharpe_ratio = 0
annualised_sharpe_ratio = 0
expected_payout = 0
rrr = 0
avg_win_streak = 0
avg_win_streak_value = 0
avg_loss_streak = 0
avg_loss_streak_value = 0
average_profit = 0
average_loss = 0
average_max_profit = 0
average_trade_duration = 0
average_time_in_profit = 0
average_time_in_drawdown = 0
max_element = 0
max_wins_ind = 0
win_streaks_value = 0
loss_streaks_value = 0
max_low_element = 0
max_loss_ind = 0
max_consecutive_profit_count = 0
max_consecutive_profit_value = 0
max_consecutive_loss_count = 0
max_consecutive_loss_value = 0
drawdown_absolute = 0
drawdown_maximal = 0
drawdown_relative = 0

def refresh_data(server, account, password, path):
    """Fetch fresh data from MT5 and process it"""
    global cached_data, last_refresh, account_info, starting_balance, mt_connected
    
    try:
        # Retry MetaTrader5 initialisation
        if retryable_initialize(3, 5, path, password, account, server):
            account_info=mt5.account_info()
            mt_connected = True

        # Define the time period for fetching deals
        from_date = datetime(2020, 1, 1)
        to_date = datetime.now() + timedelta(days=100)

        # Fetch the historical deals
        position_deals = mt5.history_deals_get(from_date, to_date)
        if position_deals is None:
            print("No deals in history.")
            print("Error code =", mt5.last_error())
            
            return None

        account_info = mt5.account_info()
        print(account_info)
        if account_info == None:
            return None

        # Convert to pandas DataFrame
        df = pd.DataFrame(list(position_deals), columns=position_deals[0]._asdict().keys())
        df['time'] = pd.to_datetime(df['time'], unit='s')
        starting_balance = (position_deals[0][13])

        # Process the data
        output_data = []
        cumulative_balance = 0

        for position_id in df['position_id'].unique():
            position_deals = df[df['position_id'] == position_id]
            if len(position_deals) < 2 and position_deals.iloc[0]['type'] != 2:
                continue

            position_deals = position_deals.sort_values(by='time')
            entry_deal = position_deals.iloc[0]
            closing_deal = position_deals.iloc[-1]
            trade_duration = (closing_deal['time'] - entry_deal['time']).total_seconds()

            output_data.append({
                'position_id': position_id,
                'symbol': entry_deal['symbol'],
                'entry_time': entry_deal['time'],
                'closing_time': closing_deal['time'],
                'entry_price': entry_deal['price'],
                'closing_price': closing_deal['price'],
                'volume': entry_deal['volume'],
                'type': entry_deal['type'],
                'magic': entry_deal['magic'],
                'swap': entry_deal['swap'] + closing_deal['swap'],
                'commission': entry_deal['commission'] + closing_deal['commission'],
                'profit': closing_deal['profit'],
                'balance': cumulative_balance + closing_deal[['commission', 'swap', 'profit', 'fee']].cumsum(axis=0)['profit'],
                'comment': entry_deal['comment'],
                'trade_duration': trade_duration
            })

            cumulative_balance += entry_deal['commission'] + entry_deal['swap'] + closing_deal['profit'] + entry_deal.get('fee', 0)

        result_df = pd.DataFrame(output_data)
        
        
        # Apply your existing calculations and transformations
        # ... (rest of your data processing logic)
        for index, row in result_df.iterrows():
            if row['type'] >= 2:
                continue

            symbol = row['symbol']
            entry_time = row['entry_time']
            close_time = row['closing_time']
            entry_price = row['entry_price']
            closing_price = row['closing_price']
            trade_type = row['type']  # 0 for buy, 1 for sell
            trade_volume = row['volume']

            profit_mins = 0
            drawdown_mins = 0
            
            # Retry initialisation and fetch price data
            retryable_initialize(3, 5, path, password, account, server)
            price_data = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, entry_time, close_time)  # Fetch price data
            symbol_info = mt5.symbol_info(symbol)
            
            for i in price_data:
                if trade_type == 0:
                    if i[1] >= entry_price:
                        profit_mins += 1
                    elif i[1] < entry_price:
                        drawdown_mins += 1
                    else:
                        drawdown_mins += 1
                elif trade_type == 1:
                    if i[1] < entry_price:
                        profit_mins += 1
                    elif i[1] > entry_price:
                        drawdown_mins += 1
                    else:
                        drawdown_mins += 1
            
            

            # Check if symbol_info is valid
            if not symbol_info:
                print(f"Error: Could not retrieve symbol info for {symbol}")
                continue
            

            # Extract necessary information from symbol_info
            tick_value = float(symbol_info.trade_tick_value)
            tick_size = float(symbol_info.trade_tick_size)
            contract_size = float(symbol_info.trade_contract_size)

            # Convert price data to DataFrame
            price_df = pd.DataFrame(price_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'extra'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='s')  # Convert timestamp to datetime
            
            # Calculate max profit based on trade type
            if trade_type == 0:  # Buy trade
                max_price_move = price_df['high'].max() - entry_price  # Price move in pips
                time_in_profit = price_df[price_df['high'] > entry_price].shape[0]
            elif trade_type == 1:  # Sell trade
                max_price_move = entry_price - price_df['low'].min()  # Price move in pips
                time_in_profit = price_df[price_df['low'] < entry_price].shape[0]
            
            # Convert price move to monetary value
            max_profit = (max_price_move / tick_size) * tick_value * trade_volume

            # Calculate trade duration
            trade_duration = (close_time - entry_time).total_seconds()
            
            # Store the results in the DataFrame
            result_df.at[index, 'max_profit'] = max_profit
            result_df.at[index, 'time_in_profit'] = time_in_profit
            result_df.at[index, 'profit_time'] = profit_mins
            result_df.at[index, 'drawdown_time'] = drawdown_mins
            result_df.at[index, 'trade_duration'] = trade_duration

        result_df['entry_price'] = result_df['entry_price'].round(2)
        result_df['closing_price'] = result_df['closing_price'].round(2)
        result_df['volume'] = result_df['volume'].round(3)
        result_df['swap'] = result_df['swap'].round(2)
        result_df['commission'] = result_df['commission'].round(2)
        result_df['profit'] = result_df['profit'].round(2)
        result_df['balance'] = result_df['balance'].round(2)
        result_df['max_profit'] = (result_df['max_profit']).round(2)

        result_df['time_in_profit'] = (result_df['time_in_profit'] / 60).round(2)  # Convert to minutes and round to 2 decimals
        result_df['trade_duration'] = (result_df['trade_duration'] / 60).round(2)  # Convert to minutes and round to 2 decimals
        result_df['profit_per_second'] = (result_df['max_profit'] / result_df['trade_duration']).round(5)
            
        
        cached_data = result_df
        last_refresh = datetime.now()
        return result_df
    
    except Exception as e:
        print(f"Error refreshing data: {str(e)}")
        
        mt_connected = False
        return None

def get_filtered_data(server, account, password, path, symbol=None, comment=None, trade_type=None, magic=None, weekday=None):
    """Get filtered data based on parameters"""
    global cached_data, starting_balance, extra_text, gross_profit, gross_loss, long_wins, short_wins, long_loss, short_loss, win_rate, largest_profit, largest_loss, profit_factor, recovery_factor, sharpe_ratio, annualised_sharpe_ratio, expected_payout, rrr, avg_win_streak, avg_win_streak_value, avg_loss_streak, avg_loss_streak_value, average_profit, average_loss, average_max_profit, average_trade_duration, average_time_in_profit, average_time_in_drawdown, max_element, max_wins_ind, win_streaks_value, loss_streaks_value, max_low_element, max_loss_ind, max_consecutive_profit_count, max_consecutive_profit_value, max_consecutive_loss_count, max_consecutive_loss_value, drawdown_absolute, drawdown_maximal, drawdown_relative

    if cached_data is None:
        refresh_data(server, account, password, path)
        if cached_data is None:
            return None
    
    filtered_df = cached_data.copy()
    extra_text = ''

    if symbol:
        filtered_df = filtered_df[filtered_df['symbol'].str.contains(symbol, na=False)]
        extra_text += f'{symbol} '
    if comment:
        filtered_df = filtered_df[filtered_df['comment'].str.contains(comment, na=False)]
        extra_text += f'{comment} '
    if trade_type is not None:
        try:
            trade_type_int = int(trade_type) if trade_type is not None else None
            filtered_df = filtered_df[filtered_df['type'] == trade_type_int]
            if trade_type_int == 0:
                extra_text += f'BUYS '
            elif trade_type_int == 1:
                extra_text += 'SELLS '
            else:
                extra_text += f'{trade_type_int} '
        except ValueError:
            pass
    if magic:
        try:
            magic_int = int(magic) if magic else None
            filtered_df = filtered_df[filtered_df['magic'] == magic_int]
            extra_text += f'{magic_int} '
        except ValueError:
            pass
    if weekday:
        filtered_df = filtered_df[filtered_df['entry_time'].dt.strftime("%A") == weekday]
        extra_text += f'{weekday} '
    
    filtered_df = filtered_df.sort_values(by='entry_time')
    filtered_df['balance'] = (filtered_df['profit'] + filtered_df['swap'] + 
                             filtered_df['commission']).cumsum()

    extra_text = f"{extra_text.strip()} " if extra_text else ''

    ########### GROSS PROFIT/LOSS #################
    gross_profit = filtered_df[filtered_df['type'] != 2][filtered_df['profit'] >= 0]['profit'].sum()
    gross_loss = abs(filtered_df[filtered_df['type'] != 2][filtered_df['profit'] < 0]['profit'].sum())
    
    ################ AVERAGE RRR ################
    profitable_trades = filtered_df[filtered_df['profit'] > 0]
    losing_trades = filtered_df[filtered_df['profit'] < 0]

    average_profit = profitable_trades['profit'].mean() if len(profitable_trades) > 0 else 0
    average_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

    if average_loss != 0:
        rrr = average_profit / abs(average_loss)
    else:
        rrr = None

    if rrr is not None:
        #print(f"Average RRR: {rrr:.2f}")
        pass
    else:
        print("Average RRR could not be calculated (no losses).")

    ################ PROFIT FACTOR ################

    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = None

    if profit_factor is not None:
        #print(f"Profit Factor: {profit_factor:.2f}")
        pass
    else:
        print("Profit Factor could not be calculated (no losses).")

    ################ SHARPE RATIO ################


    filtered_df = filtered_df[(filtered_df['volume'] > 0) & (filtered_df['profit'].notna())]
    filtered_df['entry_time'] = pd.to_datetime(filtered_df['entry_time'])
    filtered_df['entry_date'] = filtered_df['entry_time'].dt.date

    # Filter the data
    filtered_df = filtered_df[(filtered_df['volume'] > 0) & (filtered_df['profit'].notna())]
    filtered_df['entry_time'] = pd.to_datetime(filtered_df['entry_time'])
    filtered_df['entry_date'] = filtered_df['entry_time'].dt.date

    # Group by entry_date and calculate total profit per day
    daily_profit = filtered_df.groupby('entry_date')['profit'].sum().reset_index()

    # Calculate daily balance (assuming a starting balance of 10,000)

    daily_profit['balance'] = starting_balance + daily_profit['profit'].cumsum()

    # Calculate percentage increase from previous day
    daily_profit['pct_increase'] = daily_profit['balance'].pct_change()
    daily_profit['pct_increase'] = daily_profit['pct_increase'].fillna(0)  # Handle NaN values

    # Group by entry_date and aggregate
    collated_df = filtered_df.groupby('entry_date').agg({
        'position_id': 'count',  # Number of trades per day
        'profit': 'sum',         # Total profit per day
        'volume': 'sum',         # Total volume traded per day
        'swap': 'sum',           # Total swap per day
        'commission': 'sum',     # Total commission per day
        'trade_duration': 'mean',  # Average trade duration per day
        'max_profit': 'max',     # Maximum profit per day
        'time_in_profit': 'sum', # Total time in profit per day
        'profit_time': 'sum',    # Total profit time per day
        'drawdown_time': 'sum',  # Total drawdown time per day
        'profit_per_second': 'mean'  # Average profit per second per day
    }).reset_index()

    # Merge this data back into the collated_df
    collated_df = pd.merge(collated_df, daily_profit[['entry_date', 'pct_increase']], on='entry_date', how='left')

    # Rename columns for clarity
    collated_df = collated_df.rename(columns={'position_id': 'num_trades'})

    # Format percentage increase to two decimal places
    collated_df['pct_increase'] = collated_df['pct_increase']


    # Display the final DataFrame
    #print(collated_df)

    daily_returns = collated_df['profit']
    mean_daily_return = daily_returns.mean()
    #print(mean_daily_return)

    total_squared_deviation = 0
    for value in collated_df['profit']:
        squared_deviation = (value - mean_daily_return)**2
        #print(squared_deviation)
        total_squared_deviation += squared_deviation

    variance = total_squared_deviation/len(collated_df)

    portfolio_return = daily_returns.sum() / len(daily_returns)
    risk_free_rate = 0.045 / 252
    standard_deviation = math.sqrt(variance)

    sharpe_ratio = (portfolio_return - risk_free_rate)/ standard_deviation
    annualised_sharpe_ratio = sharpe_ratio * math.sqrt(252)

    #print(f'Sharpe Ratio: {sharpe_ratio}')
    #print(f'Annualised Sharpe Ratio: {annualised_sharpe_ratio}')


    std_dev_daily_return = daily_returns.std()

    ################ RECOVERY FACTOR ##################

    # Total Net Profit
    total_net_profit = filtered_df[filtered_df['type'] != 2]['profit'].sum()

    # Calculate Maximum Drawdown (Absolute)
    balance = filtered_df['balance']
    running_max = balance.cummax()  # Running maximum of the balance
    drawdown = running_max - balance  # Drawdown at each point
    max_drawdown_absolute = drawdown.max()  # Maximum drawdown

    # Calculate Recovery Factor
    if max_drawdown_absolute > 0:
        recovery_factor = total_net_profit / max_drawdown_absolute
    else:
        recovery_factor = None  # Avoid division by zero

    # Display Recovery Factor
    if recovery_factor is not None:
        #print(f"Recovery Factor: {recovery_factor:.2f}")
        pass
    else:
        print("Recovery Factor could not be calculated (no drawdown).")



    ########### EXPECTED PAYOUT ###################

    total_net_profit = filtered_df[filtered_df['type'] != 2]['profit'].sum()

    total_trades = len(filtered_df)


    # Calculate Expected Payout
    if total_trades > 0:
        expected_payout = total_net_profit / total_trades
    else:
        expected_payout = None  # Avoid division by zero

    # Display the Expected Payout
    if expected_payout is not None:
        #print(f"Expected Payout: {expected_payout:.2f}")
        pass
    else:
        print("Expected Payout could not be calculated (no trades).")

    ############# BALANCE DRAWDOWN ###################

    running_max = filtered_df['balance'].cummax()    # Running peak balance
    drawdown = running_max - filtered_df['balance']  # Drawdown at each point
    lowest_balance = filtered_df['balance'].min()
    drawdown_absolute = starting_balance - lowest_balance

    drawdown_maximal = drawdown.max()

    # Calculate Drawdown Relative
    if running_max.max() > 0:
        drawdown_relative = (drawdown_maximal / running_max.max()) * 100
    else:
        drawdown_relative = None  # Avoid division by zero

    if drawdown_relative is not None:
        #print(f"Balance Drawdown Relative: {drawdown_relative:.2f}%")
        pass
    else:
        print("Balance Drawdown Relative could not be calculated (no peak balance).")
    
    ####################### LONG/SHORT TRADES ######################

    # Filter buy (long) and sell (short) trades
    long_trades = filtered_df[filtered_df['type'] == 0]  # Buy (long) trades
    short_trades = filtered_df[filtered_df['type'] == 1]  # Sell (short) trades
    print(len(long_trades))
    print(len(short_trades))

    # Calculate winning long trades (long trades are won when closing_price > entry_price)
    long_wins = len(long_trades[long_trades['profit'] >= 0 ])
    long_loss = len(long_trades) - long_wins
    long_win_percentage = (long_wins) / len(long_trades) * 100 if len(long_trades) > 0 else 0

    # Calculate winning short trades (short trades are won when closing_price < entry_price)
    short_wins = len(short_trades[short_trades['profit'] >= 0 ])
    short_loss = len(short_trades) - short_wins
    short_win_percentage = (short_wins) / len(short_trades) * 100 if len(short_trades) > 0 else 0

    # Print the results
    #print(f'Long Trades (won %): {long_win_percentage:.2f}')
    #print(f'Short Trades (won %): {short_win_percentage:.2f}')

    ################### PROFIT TRADES #############################

    total_profitable_trades = short_wins + long_wins

    total_loss_trades = len(filtered_df) - total_profitable_trades

    profit_trade_percent = (total_profitable_trades) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    loss_trade_percent = (total_loss_trades) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0

    ################### LARGEST PROFIT/ LOSS ######################

    largest_profit = filtered_df[filtered_df['type'] != 2]['profit'].max()
    largest_loss = filtered_df[filtered_df['type'] != 2]['profit'].min()

    ################## AVERAGE PROFIT/LOSS ######################

    profitable_trades = filtered_df[filtered_df['type'] != 2][filtered_df['profit'] >= 0]
    losing_trades = filtered_df[filtered_df['type'] != 2][filtered_df['profit'] < 0]



    # Calculate the average profit and average loss
    average_profit = profitable_trades['profit'].mean() if len(profitable_trades) > 0 else 0
    average_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

    # Print the results
    #print(f'Average profit trade: {average_profit:.2f}')
    #print(f'Average loss trade: {average_loss:.2f}')

    ################# CONSECUTIVE WIN/LOSS #######################

    # Variables to track the maximum consecutive wins and losses (count and value)
    max_consecutive_wins_count = 0
    max_consecutive_losses_count = 0
    max_consecutive_wins_value = 0.0
    max_consecutive_losses_value = 0.0

    current_wins_count = 0
    current_losses_count = 0
    current_wins_value = 0.0
    current_losses_value = 0.0

    # Iterate over the 'profit' column to track consecutive wins and losses
    for profit in filtered_df['profit']:
        if profit >= 0:  # Win
            current_wins_count += 1  # Increment the win count
            current_wins_value += profit  # Accumulate the profit value of consecutive wins
            
            # Reset the losses counter
            current_losses_count = 0
            current_losses_value = 0.0

            
            
            # Update the maximum consecutive wins count and value
            if current_wins_count > max_consecutive_wins_count:
                max_consecutive_wins_count = current_wins_count
                max_consecutive_wins_value = current_wins_value
        
        elif profit < 0:  # Loss
            current_losses_count += 1  # Increment the loss count
            current_losses_value += profit  # Accumulate the loss value of consecutive losses
            
            # Reset the wins counter
            current_wins_count = 0
            current_wins_value = 0.0
            
            # Update the maximum consecutive losses count and value
            if current_losses_count > max_consecutive_losses_count:
                max_consecutive_losses_count = current_losses_count
                max_consecutive_losses_value = current_losses_value

        #print(f'consecutive wins: {current_wins_count}')
        #print(f'consecutive losses: {current_losses_count}')
        #print(f'max consecutive wins: {max_consecutive_wins_count}')
        #print(f'max consecutive losses: {max_consecutive_losses_count}')

    ################### MAXIMAL CONSECUTIVE PROFIT/LOSS #####################

    # Initialize variables to track the max streaks and their respective values
    max_consecutive_profit_count = 0
    max_consecutive_profit_value = 0.0
    max_consecutive_loss_count = 0
    max_consecutive_loss_value = 0.0

    current_wins_count = 0
    current_losses_count = 0
    current_wins_value = 0.0
    current_losses_value = 0.0

    # Variables to calculate averages
    total_wins_count = 0
    total_losses_count = 0
    total_wins_streaks = 0
    total_losses_streaks = 0

    win_streaks = []
    loss_streaks = []
    win_streaks_value = []
    loss_streaks_value = []
    # Iterate over the profit values
    for profit in filtered_df[filtered_df['type'] != 2]['profit']:
        if profit > 0:  # Win
            current_wins_count += 1
            current_wins_value += profit

            # Reset losses if there was a loss streak
            if current_losses_count > 1:
                loss_streaks.append(current_losses_count)
                loss_streaks_value.append(current_losses_value)
                total_losses_streaks += 1
                total_losses_count += current_losses_count
                current_losses_count = 0
                current_losses_value = 0.0

            # Track maximal consecutive wins
            if current_wins_count > max_consecutive_profit_count:
                max_consecutive_profit_count = current_wins_count
                max_consecutive_profit_value = current_wins_value

        elif profit < 0:  # Loss
            current_losses_count += 1
            current_losses_value += profit

            # Reset wins if there was a win streak
            if current_wins_count > 1:
                win_streaks.append(current_wins_count)
                win_streaks_value.append(current_wins_value)
                total_wins_streaks += 1
                total_wins_count += current_wins_count
                current_wins_count = 0
                current_wins_value = 0.0

            # Track maximal consecutive losses
            if current_losses_count > max_consecutive_loss_count:
                max_consecutive_loss_count = current_losses_count
                max_consecutive_loss_value = current_losses_value

    # Append any ongoing streak at the end of the loop
    if current_wins_count > 0:
        win_streaks.append(current_wins_count)
        win_streaks_value.append(current_wins_value)
        total_wins_streaks += 1
        total_wins_count += current_wins_count

    if current_losses_count > 0:
        loss_streaks.append(current_losses_count)
        loss_streaks_value.append(current_losses_value)
        total_losses_streaks += 1
        total_losses_count += current_losses_count

    # Calculate averages (with checks for empty lists)
    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

    avg_win_streak_value = sum(win_streaks_value) / len(win_streaks_value) if win_streaks_value else 0
    avg_loss_streak_value = sum(loss_streaks_value) / len(loss_streaks_value) if loss_streaks_value else 0
    max_wins_ind = 0
    max_element = win_streaks[0]

    for i in range (1,len(win_streaks)): #iterate over array
        if win_streaks[i] > max_element: #to check max value
            max_element = win_streaks[i]
            max_wins_ind = i

    max_loss_ind = 0
    max_low_element = loss_streaks[0]

    for i in range (1,len(loss_streaks)): #iterate over array
        if loss_streaks[i] > max_low_element: #to check max value
            max_low_element = loss_streaks[i]
            max_loss_ind = i
    
    average_consecutive_wins = avg_win_streak
    average_consecutive_losses = avg_loss_streak

    ####################### AVERAGE MAX PROFIT ###########################

    # Calculate average values
    average_max_profit = filtered_df['max_profit'].mean()  # Average monetary max profit
    average_time_in_profit = filtered_df['profit_time'].mean()  # Average time in profit
    average_time_in_drawdown = filtered_df['drawdown_time'].mean()  # Average time in profit
    average_trade_duration = filtered_df['trade_duration'].mean()  # Average trade duration in seconds


    ################ WIN RATE ################

    win_rate = ((long_wins + short_wins)/((long_wins + short_wins) + (long_loss + short_loss))) * 100

    #################################################################################
    



    print(filtered_df)
    return filtered_df


empty_df = pd.DataFrame({
    'position_id': [0, 1], 
    'symbol': ['one', 'two'], 
    'entry_time': [0, 1], 
    'closing_time': [2,3], 
    'entry_price': [10, 20], 
    'closing_price': [30, 10], 
    'volume': [1, 2], 
    'type': [0 , 1], 
    'magic': ['', ''], 
    'swap': [0, 0], 
    'commission': [0,0], 
    'profit': [300, 400], 
    'balance': [100300, 100700], 
    'comment': ['test', 'tube'], 
    'trade_duration': [100, 100]
})

# Create empty figures for initial layout
empty_pie = px.pie(empty_df, names='symbol', hole=0.7, labels='profit')
empty_hist = px.histogram(empty_df, x='symbol', y='profit', barmode='group', color='type',  color_discrete_map={0: '#636efa', 1: '#ef553b'})
empty_line = px.line(empty_df, x='entry_time', y='balance')
empty_donut = px.pie(
    pd.DataFrame({'type': ['Gross Profit', 'Gross Loss'], 'amount': [700, 0]}),
    values='amount', names='type', hole=0.6,
    color_discrete_map={'Gross Profit': '#2ca02c', 'Gross Loss': '#d62728'}
)


external_stylesheets = [dbc.themes.LUMEN]
app = Dash(__name__, external_stylesheets=external_stylesheets)


# Define the layout with filter controls
app.layout = dbc.Container([

    
    dbc.Row([
        html.Div('Trading Analytics Dashboard', className="text-primary text-center fs-3")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                id='connection-status',
                children="Not connected to MT5",
                color="danger",
                className="mb-4"
            )
        ])
    ]),
    
    # MT5 Credentials Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("MT5 Connection Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Server"),
                            dbc.Input(id='server-input', type='text', value=mt_server),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Account"),
                            dbc.Input(id='account-input', type='number', value=mt_account),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Password"),
                            dbc.Input(id='password-input', type='password', value=mt_pass),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Terminal Path"),
                            dbc.Input(id='path-input', type='text', value=terminal_path),
                        ], width=3),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Connect & Refresh Data", id='connect-button', color="primary", className="mt-2"),
                        ], width=12),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    
    # Data Filters Section
    dbc.Row(id='data-filters-row', style={'display': 'none'}, children=[
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade Data Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol"),
                            dbc.Input(id='symbol-filter', type='text', placeholder='Filter by symbol'),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Comment"),
                            dbc.Input(id='comment-filter', type='text', placeholder='Filter by comment'),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Trade Type"),
                            dcc.Dropdown(
                                id='type-filter',
                                options=[
                                    {'label': 'All Trades', 'value': 'all'},
                                    {'label': 'BUYS', 'value': '0'},
                                    {'label': 'SELLS', 'value': '1'},
                                ],
                                value='all',
                                clearable=False
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Magic Number"),
                            dbc.Input(id='magic-filter', type='number', placeholder='Filter by magic number'),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Weekday"),
                            dcc.Dropdown(
                                id='weekday-filter',
                                options=[
                                    {'label': 'All Days', 'value': 'all'},
                                    {'label': 'Monday', 'value': 'Monday'},
                                    {'label': 'Tuesday', 'value': 'Tuesday'},
                                    {'label': 'Wednesday', 'value': 'Wednesday'},
                                    {'label': 'Thursday', 'value': 'Thursday'},
                                    {'label': 'Friday', 'value': 'Friday'},
                                    {'label': 'Saturday', 'value': 'Saturday'},
                                    {'label': 'Sunday', 'value': 'Sunday'},
                                ],
                                value='all',
                                clearable=False
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Apply Filters"),
                            dbc.Button("Filter Data", id='filter-button', color="primary", className="mt-2"),
                        ], width=2),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    
    # Main Dashboard Content
    dbc.Row(id='dashboard-content', children=[
        dbc.Col([
            dcc.Loading( 
                dcc.Graph(id='symbol-pie', figure=empty_pie)
            )
        ], width=5, style={"height": "500px"}, className="pe-2"),
        dbc.Col([ 
            dcc.Graph(id='profit-histogram', figure=empty_hist)
        ], width=4, style={"height": "500px"}, className="pe-2"),
        dbc.Col([
            html.Div(id='metrics-container', children=[
                dbc.Row([ 
                    html.H6("Profit Factor", className="card-title"),
                    dbc.Progress(id='profit-factor', label=round(profit_factor, 2), value=(profit_factor/5)*100)
                ]),
                dbc.Row([ 
                    html.H6("Recovery Factor", className="card-title"),
                    dbc.Progress(id='recovery-factor', label=round(recovery_factor, 2), value=(recovery_factor/5)*100)
                ]),
                dbc.Row([ 
                    html.H6("Expected Pay", className="card-title"),
                    dbc.Progress(id='expected-payout', label=round(expected_payout, 2), value=(expected_payout))
                ]),
                dbc.Row([ 
                    html.H6("Annualised Sharpe Ratio"),
                    dbc.Progress(id='annualised-sharpe', label=round(annualised_sharpe_ratio, 2), value=(annualised_sharpe_ratio/5)*100)
                ]),
                dbc.Row([ 
                    html.H6("Sharpe Ratio", className="card-title"),
                    dbc.Progress(id='sharpe-ratio', label=round(sharpe_ratio, 2), value=(sharpe_ratio/2)*100, className="no-left-radius"),
                ]),
            ], style={"height": "100%", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
        ], width=3, className="pe-2", style={"height": "500px"})  
    ],
     style={
        "marginBottom": "20px",
        "alignItems": "stretch",
        "display": "flex !important",
        "flex-wrap": "nowrap !important",
        "overflow-x": "auto"
    },
      ),
    
    dbc.Row([
        dbc.Col([ 
            dcc.Graph(id='balance-chart', figure=empty_line)
        ], width=8),
        dbc.Col([ 
            dcc.Graph(id='profit-donut', figure=empty_donut)
        ], width=4)
    ]),

    dbc.Row([
        dbc.Col([ 
            dash_table.DataTable(
                id='data-table',
                data=empty_df.to_dict('records'), 
                page_size=10, 
                style_table={'overflowX': 'auto'},
            )
            
        ])
    ]),
    
    # Store for filtered data
    dcc.Store(id='filtered-data-store'),
])


# Callback for MT5 connection
@app.callback(
    [
        Output('connection-status', 'children'),
        Output('connection-status', 'color'),
        Output('data-filters-row', 'style'),
        Output('dashboard-content', 'style'),
        Output('filtered-data-store', 'data')
    ],
    Input('connect-button', 'n_clicks'),
    [
        State('server-input', 'value'),
        State('account-input', 'value'),
        State('password-input', 'value'),
        State('path-input', 'value')
    ],
    prevent_initial_call=True
)
def connect_to_mt5(n_clicks, server, account, password, path):
    if not all([server, account, password, path]):
        return [
            "Please fill in all connection details",
            "danger",
            {'display': 'none'},
            {'display': 'none'},
            None
        ]
    
    try:
        df = refresh_data(server, account, password, path)
        if df is not None:
            return [
                f"Connected to MT5 (Account: {account})",
                "success",
                {'display': 'block'},
                {'display': 'flex'},
                df.to_dict('records')
            ]
        else:
            return [
                "Failed to connect to MT5 - check credentials",
                "danger",
                {'display': 'none'},
                {'display': 'none'},
                None
            ]
    except Exception as e:
        return [
            f"Connection error: {str(e)}",
            "danger",
            {'display': 'none'},
            {'display': 'none'},
            None
        ]
    
@app.callback(
    [
        Output('filtered-data-store', 'data', allow_duplicate=True),
        Output('symbol-pie', 'figure'),
        Output('profit-histogram', 'figure'),
        Output('balance-chart', 'figure'),
        Output('profit-donut', 'figure'),
        Output('data-table', 'data'),
        Output('profit-factor', 'value'),
        Output('profit-factor', 'label'),
        Output('recovery-factor', 'value'),
        Output('recovery-factor', 'label'),
        Output('expected-payout', 'value'),
        Output('expected-payout', 'label'),
        Output('annualised-sharpe', 'value'),
        Output('annualised-sharpe', 'label'),
        Output('sharpe-ratio', 'value'),
        Output('sharpe-ratio', 'label'),
    ],
    Input('filter-button', 'n_clicks'), 
    Input('connect-button', 'n_clicks'),
    [
        State('server-input', 'value'),
        State('account-input', 'value'),
        State('password-input', 'value'),
        State('path-input', 'value'),
        State('symbol-filter', 'value'),
        State('comment-filter', 'value'),
        State('type-filter', 'value'),
        State('magic-filter', 'value'),
        State('weekday-filter', 'value'),
        State('filtered-data-store', 'data')
    ],
    prevent_initial_call=True
)
def apply_filters(filter_clicks, connect_clicks, server, account, password, path, 
                 symbol, comment, trade_type, magic, weekday, current_data):
    ctx = callback_context
    
    if not ctx.triggered:
        raise exceptions.PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If connect button was clicked, apply no filters (show all data)
    if trigger_id == 'connect-button':
        symbol = None
        comment = None
        trade_type = None
        magic = None
        weekday = None

    if not mt_connected:
        raise exceptions.PreventUpdate
    
    # Convert trade_type to None if 'all' is selected
    trade_type = int(trade_type) if trade_type != 'all' else None
    # Convert weekday to None if 'all' is selected
    weekday = weekday if weekday != 'all' else None
    
    # Apply filters
    filtered_df = get_filtered_data(
        server=server,
        account=account,
        password=password,
        path=path,
        symbol=symbol,
        comment=comment,
        trade_type=trade_type,
        magic=magic,
        weekday=weekday
    )
    
    if filtered_df is None or len(filtered_df) == 0:
        return [
            empty_df.to_dict('records'),
            empty_pie,
            empty_hist,
            empty_line,
            empty_donut,
            empty_df.to_dict('records'),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Default values for metrics
        ]
    
    # Update profit/loss donut chart
    gross_profit = filtered_df[filtered_df['type'] != 2][filtered_df['profit'] >= 0]['profit'].sum()
    gross_loss = abs(filtered_df[filtered_df['type'] != 2][filtered_df['profit'] < 0]['profit'].sum())
    net_profit = gross_profit - gross_loss
    
    profit_df = pd.DataFrame({
        'type': ['Gross Profit', 'Gross Loss'],
        'amount': [gross_profit, gross_loss]
    })
    
    donut_fig = px.pie(profit_df, values='amount', names='type', hole=0.6,
                      color_discrete_map={'Gross Profit': '#2ca02c', 'Gross Loss': '#d62728'})
    
    donut_fig.update_layout(
        annotations=[dict(text=f'Net Profit: {round(net_profit, 2)}', x=0.5, y=0, font_size=14, showarrow=False)]
    )
    
    # Return all outputs
    return [
        filtered_df.to_dict('records'),
        px.pie(filtered_df, names='symbol', hole=.7, labels='profit'),
        px.histogram(
            filtered_df, 
            x='symbol', 
            y='profit', 
            barmode='group', 
            color='type',
            color_discrete_map={0: '#636efa', 1: '#ef553b'},
        ).update_layout(
            legend_title_text='Transaction Type',
            legend=dict(
                title_font=dict(size=12),
                itemsizing='constant'
            )
        ).update_traces(
            selector=dict(),
            name='BUYS',
            hovertemplate='BUYS: %{y}<extra></extra>'
        ).update_traces(
            selector=dict(legendgroup='1'),
            name='SELLS',
            hovertemplate='SELLS: %{y}<extra></extra>'
        ),
        px.line(filtered_df, x='entry_time', y='balance'),
        donut_fig,
        filtered_df.to_dict('records'),
        (profit_factor/5)*100,
        round(profit_factor, 2),
        (recovery_factor/5)*100,
        round(recovery_factor, 2),
        expected_payout,
        round(expected_payout, 2),
        (annualised_sharpe_ratio/5)*100,
        round(annualised_sharpe_ratio, 2),
        (sharpe_ratio/2)*100,
        round(sharpe_ratio, 2),
    ]

if __name__ == '__main__':
    app.run(debug=True)
