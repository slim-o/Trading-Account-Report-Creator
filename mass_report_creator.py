from func_general_functions import *
from variables_general import *
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdfkit
import argparse
import os
import math


pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 20000)     # max table width to display
pd.set_option('display.max_rows', None)

# Retry MetaTrader5 initialisation
retryable_initialize(3, 5, terminal_path)

# Define the time period for fetching deals
from_date = datetime(2020, 1, 1)
to_date = datetime.now() + timedelta(days=1)

# Fetch the historical deals
position_deals = mt5.history_deals_get(from_date, to_date)
if position_deals is None:
    print("No deals in history.")
    print("Error code =", mt5.last_error())
    mt5.shutdown()
    quit()
#else:
#    print(position_deals)

account_info=mt5.account_info()
if account_info == None:
    quit()

#print(account_info)



# Convert to pandas DataFrame
df = pd.DataFrame(list(position_deals), columns=position_deals[0]._asdict().keys())
df['time'] = pd.to_datetime(df['time'], unit='s')
#print(position_deals)
#print(df)
starting_balance = (position_deals[0][13])

# Group by position_id and calculate entry and closing prices
output_data = []
cumulative_balance = 0

for position_id in df['position_id'].unique():
    position_deals = df[df['position_id'] == position_id]
    #print(position_deals.iloc[0]['type'])
    #print('')
    if len(position_deals) < 2 and position_deals.iloc[0]['type'] != 2:
        continue  # Skip if there's no closing deal
    

    position_deals = position_deals.sort_values(by='time')
    entry_deal = position_deals.iloc[0]  # Entry deal
    closing_deal = position_deals.iloc[-1]  # Closing deal

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
        'trade_duration': trade_duration  # New column
    })

    cumulative_balance += entry_deal['commission'] + entry_deal['swap'] + closing_deal['profit'] + entry_deal.get('fee', 0)

# Convert aggregated data to a DataFrame
result_df = pd.DataFrame(output_data)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate filtered trading reports.")
    
    # Define arguments
    parser.add_argument("--symbol", type=str, help="Filter by symbol (e.g., EURUSD)", required=True)
    parser.add_argument("--trade_type", type=int, help="Filter by trade type: 0 for buys, 1 for sells", choices=[0, 1], required=False)
    parser.add_argument("--comment", type=str, help="Filter by comment (e.g., V4.1_T1_daily)", required=False)
    parser.add_argument('--weekday', type=str, help='Filter trades by weekday (e.g., Monday, Tuesday).')

    # Parse the arguments
    args = parser.parse_args()
    return args

def get_output_directory(base_dir, symbol, trade_type=None, comment=None, weekday=None):
    dir_path = os.path.join(base_dir, symbol)
    if trade_type == 0:
        dir_path = os.path.join(dir_path, "Buys")
    elif trade_type == 1:
        dir_path = os.path.join(dir_path, "Sells")
    elif comment:
        dir_path = os.path.join(dir_path, "Thirds")
    elif weekday:
        dir_path = os.path.join(dir_path, weekday)


    os.makedirs(dir_path, exist_ok=True)
    return dir_path

extra_text = ''

def filter_dataframe(result_df, symbol=None, comment=None, trade_type=None, magic=None, weekday=None):
    global extra_text  # Declare to modify the global variable
    extra_text = ''  # Reset the global variable at the start of the function
    
    filtered_df = result_df.copy()

    if symbol:
        filtered_df = filtered_df[filtered_df['symbol'] == symbol]
        extra_text += f'{symbol} '
    if comment:
        filtered_df = filtered_df[filtered_df['comment'].str.contains(comment, na=False)]
        extra_text += f'{comment} '
    if trade_type is not None:
        filtered_df = filtered_df[filtered_df['type'] == trade_type]
        if trade_type == 0:
            extra_text += f'BUYS '
        elif trade_type == 1:
            extra_text += 'SELLS '
        else:
            extra_text += f'{trade_type} '
    if magic:
        filtered_df = filtered_df[filtered_df['magic'] == magic]
        extra_text += f'{magic} '
    if weekday:
        filtered_df = filtered_df[filtered_df['entry_time'].dt.strftime("%A") == weekday]
        extra_text += f'{weekday} '
    
    # Recalculate the cumulative balance after filtering
    filtered_df = filtered_df.sort_values(by='entry_time')  # Ensure chronological order
    filtered_df['balance'] = (filtered_df['profit'] + filtered_df['swap'] + 
                              filtered_df['commission']).cumsum()  # Compute cumulative sum


    extra_text = f"{extra_text.strip()}_" if extra_text else ''  # Format extra_text
    return filtered_df

args = parse_arguments()
filtered_by_symbol = filter_dataframe(result_df, symbol=args.symbol, comment=args.comment, trade_type=args.trade_type, magic=None, weekday=args.weekday)
base_directory = f"LUCI Report {from_date.strftime('%Y-%m-%d')} - {to_date.strftime('%Y-%m-%d')}"
output_directory = get_output_directory(
    base_directory, 
    symbol=args.symbol, 
    trade_type=args.trade_type, 
    comment=args.comment,
    weekday=args.weekday

)

result_df = filtered_by_symbol

#print(result_df)

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
    retryable_initialize(3, 5, terminal_path)
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
    
    #print(f'Mins in Profit: {profit_mins}')
    #print(f'Mins in drawdown: {drawdown_mins}')

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


# Calculate totals for swap, commission, and profit
totals = {
    'position_id': 'TOTAL',
    'symbol': '',
    'entry_time': '',
    'closing_time': '',
    'entry_price': '',
    'closing_price': '',
    'volume': '',
    'type': '',
    'magic': '',
    'swap': round((result_df['swap'].sum()), 2),
    'commission': round((result_df['commission'].sum()), 2),
    # Exclude rows with type == 2 from total profit
    'profit': round((result_df[result_df['type'] != 2]['profit'].sum()), 2),
    'balance': round((result_df['swap'].sum() + result_df['commission'].sum() + result_df['profit'].sum()), 2),
    'comment': '',
    'trade_duration': '',  # Not relevant for totals
    'max_profit': '',
    'time_in_profit': '',
    'profit_time': result_df['profit_time'].sum(),
    'drawdown_time': result_df['drawdown_time'].sum(),
    'profit_per_second': ''
}

# Append totals to the DataFrame
totals_df = pd.DataFrame([totals])


# Display the final DataFrame
result_df['entry_price'] = result_df['entry_price'].round(2)
result_df['closing_price'] = result_df['closing_price'].round(2)
result_df['volume'] = result_df['volume'].round(3)
result_df['swap'] = result_df['swap'].round(2)
result_df['commission'] = result_df['commission'].round(2)
result_df['profit'] = result_df['profit'].round(2)
result_df['balance'] = result_df['balance'].round(2)

#print(result_df)
result_df['max_profit'] = (result_df['max_profit']).round(2)

# Convert time-based columns to minutes
result_df['time_in_profit'] = (result_df['time_in_profit'] / 60).round(2)  # Convert to minutes and round to 2 decimals
result_df['trade_duration'] = (result_df['trade_duration'] / 60).round(2)  # Convert to minutes and round to 2 decimals
# Calculate profit/time ratio (optional)
result_df['profit_per_second'] = (result_df['max_profit'] / result_df['trade_duration']).round(5)

# Display the formatted DataFrame
#print(result_df)
new_df = pd.concat([result_df, totals_df], ignore_index=True)
#print(new_df)

########### GROSS PROFIT/LOSS #################

gross_profit = result_df[result_df['type'] != 2][result_df['profit'] >= 0]['profit'].sum()

# Calculate Gross Loss (absolute sum of negative profits)
gross_loss = abs(result_df[result_df['type'] != 2][result_df['profit'] < 0]['profit'].sum())


################ AVERAGE RRR ################

profitable_trades = result_df[result_df['profit'] > 0]
losing_trades = result_df[result_df['profit'] < 0]

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


filtered_df = result_df[(result_df['volume'] > 0) & (result_df['profit'].notna())]
filtered_df['entry_time'] = pd.to_datetime(filtered_df['entry_time'])
filtered_df['entry_date'] = filtered_df['entry_time'].dt.date

# Filter the data
filtered_df = result_df[(result_df['volume'] > 0) & (result_df['profit'].notna())]
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

#print(filtered_df.iloc[0])


#print(daily_returns.sum())
#print(len(daily_returns))

#print(mean_daily_return)
#print(std_dev_daily_return)

portfolio_return = daily_returns.sum() / len(daily_returns)
risk_free_rate = 0.045 / 252
standard_deviation = math.sqrt(variance)

#print(f'Portfolo Return: {portfolio_return}')
#print(f'Risk Free Rate: {risk_free_rate}')
#print(f'Standard Deviation: {standard_deviation}')


sharpe_ratio = (portfolio_return - risk_free_rate)/ standard_deviation
annualised_sharpe_ratio = sharpe_ratio * math.sqrt(252)

#print(f'Sharpe Ratio: {sharpe_ratio}')
#print(f'Annualised Sharpe Ratio: {annualised_sharpe_ratio}')


std_dev_daily_return = daily_returns.std()


################ RECOVERY FACTOR ##################

# Total Net Profit
total_net_profit = result_df[result_df['type'] != 2]['profit'].sum()

# Calculate Maximum Drawdown (Absolute)
balance = result_df['balance']
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

total_net_profit = result_df[result_df['type'] != 2]['profit'].sum()

total_trades = len(result_df)


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

# Assuming `balance` column exists in the DataFrame
initial_balance = new_df['balance'].iloc[0]  # First balance (initial deposit)
#initial_balance = starting_balance
#print(initial_balance)
running_max = new_df['balance'].cummax()    # Running peak balance
#print(running_max)
drawdown = running_max - new_df['balance']  # Drawdown at each point
#print(drawdown)

# Calculate Drawdown Absolute
lowest_balance = new_df['balance'].min()
#print(lowest_balance)
drawdown_absolute = initial_balance - lowest_balance

# Calculate Drawdown Maximal
drawdown_maximal = drawdown.max()

# Calculate Drawdown Relative
if running_max.max() > 0:
    drawdown_relative = (drawdown_maximal / running_max.max()) * 100
else:
    drawdown_relative = None  # Avoid division by zero

# Print Results
#print(f"Balance Drawdown Absolute: {drawdown_absolute:.2f}")
#print(f"Balance Drawdown Maximal: {drawdown_maximal:.2f}")
if drawdown_relative is not None:
    #print(f"Balance Drawdown Relative: {drawdown_relative:.2f}%")
    pass
else:
    print("Balance Drawdown Relative could not be calculated (no peak balance).")

####################### LONG/SHORT TRADES ######################

# Filter buy (long) and sell (short) trades
long_trades = new_df[new_df['type'] == 0]  # Buy (long) trades
short_trades = new_df[new_df['type'] == 1]  # Sell (short) trades
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

total_loss_trades = len(new_df) - total_profitable_trades

profit_trade_percent = (total_profitable_trades) / len(new_df) * 100 if len(new_df) > 0 else 0
loss_trade_percent = (total_loss_trades) / len(new_df) * 100 if len(new_df) > 0 else 0

################### LARGEST PROFIT/ LOSS ######################

largest_profit = new_df[new_df['type'] != 2]['profit'].max()
largest_loss = new_df[new_df['type'] != 2]['profit'].min()

################## AVERAGE PROFIT/LOSS ######################

profitable_trades = new_df[new_df['type'] != 2][new_df['profit'] >= 0]
losing_trades = new_df[new_df['type'] != 2][new_df['profit'] < 0]



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
for profit in new_df['profit']:
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


# Print the results (in GBP for wins and USD for losses)
'''
print(f'Maximum consecutive wins (count): {max_consecutive_wins_count}')
print(f'Maximum consecutive wins (£): {max_consecutive_wins_value:.2f}')
print(f'Maximum consecutive losses (count): {max_consecutive_losses_count}')
print(f'Maximum consecutive losses ($): {max_consecutive_losses_value:.2f}')
'''
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
for profit in new_df[new_df['type'] != 2]['profit']:
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
'''
print("Win streaks:", win_streaks)
print("Loss streaks:", loss_streaks)
print("Win streaks (value):", win_streaks_value)
print("Loss streaks (value):", loss_streaks_value)
print('')
print("Average win streak:", avg_win_streak)
print("Average loss streak:", avg_loss_streak)
print("Average Win streaks (value):", avg_win_streak_value)
print("Average Loss streaks (value):", avg_loss_streak_value)

'''


average_consecutive_wins = avg_win_streak
average_consecutive_losses = avg_loss_streak

# Print the results
'''
print(f'Maximal consecutive profit (count): {max_consecutive_profit_count} {max_consecutive_profit_value:.2f}')
print(f'Maximal consecutive loss (count): {max_consecutive_loss_count} {max_consecutive_loss_value:.2f}')
print(f'Average consecutive wins: {average_consecutive_wins:.2f}')
print(f'Average consecutive losses: {average_consecutive_losses:.2f}')
'''
####################### AVERAGE MAX PROFIT ###########################

# Calculate average values
average_max_profit = result_df['max_profit'].mean()  # Average monetary max profit
average_time_in_profit = result_df['profit_time'].mean()  # Average time in profit
average_time_in_drawdown = result_df['drawdown_time'].mean()  # Average time in profit
average_trade_duration = result_df['trade_duration'].mean()  # Average trade duration in seconds

# Output the averages
#print(f"Average Maximum Profit: {average_max_profit:.2f} (in monetary value)")
#print(f"Average Time in Profit: {average_time_in_profit:.2f} (number of price data points)")
#print(f"Average Trade Duration: {average_trade_duration:.2f} seconds")

################ WIN RATE ################

win_rate = ((long_wins + short_wins)/((long_wins + short_wins) + (long_loss + short_loss))) * 100
########################################

#########
#########
#########
print('')
print('')
print(f'Name: {account_info[24]}')
print(f'Account: {account_info[0]}')
print(f'Company: {account_info[27]}')
print(f'Date: {datetime.now()}')
print('')

print('')
print(f'Balance: {account_info[10]}')
print(f'Starting Balance: {starting_balance}')
print(f'Credit Facility: {account_info[11]}')
print(f'Floating P/L: {account_info[12]}')
print(f'Equity: {account_info[13]}')
print('')

print(f'Free Margin: {account_info[15]}')
print(f'Margin: {account_info[14]}')
print(f'Margin Level: {account_info[16]}')
print('')
print('')
print(f"Total Net Profit: {(result_df[result_df['type'] != 2]['profit'].sum()):.2f}")
print(f'Gross Profit: {gross_profit:.2f}')
print(f'Gross Loss: {gross_loss:.2f}')
print('')


##########
##########

# TOTAL TRADES
print(f'Total Trades: {len(result_df)}')
print(f'Total Won: {long_wins + short_wins}')
print(f'Total Lost: {long_loss + short_loss}')
print(f'Win Rate: {win_rate:.2f}%')
print('')

print(f'Long Trades Won: {long_wins}')
print(f'Long Trades Lost: {long_loss}')
print(f'Short Trades Won: {short_wins}')
print(f'Short Trades Lost: {short_loss}')
print('')


print(f'Profit Factor: {profit_factor:.2f}')
print(f'Annualised Sharpe Ratio: {annualised_sharpe_ratio}')
print(f'Daily Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Recovery Factor: {recovery_factor:.2f}')
print(f'Expected Payoff: {expected_payout:.2f}')
print('')

#AVERAGE TRADES
print(f'Average RRR: {rrr:.2f}')
print(f'Average profit trade: {average_profit:.2f}')
print(f'Average loss trade: {average_loss:.2f}')
print("Average win streak:", avg_win_streak)
print("Average loss streak:", avg_loss_streak)
print("Average Win streaks (value):", avg_win_streak_value)
print("Average Loss streaks (value):", avg_loss_streak_value)
print('')
print("Average Max Profit (Monetary): £{:.2f}".format(average_max_profit))
print("Average Time in Profit (Minutes): {:.2f}".format(average_time_in_profit / 60))
print("Average Time in Drawdown (Minutes): {:.2f}".format(average_time_in_drawdown / 60))
print("Average Trade Duration (Minutes): {:.2f}".format(average_trade_duration / 60))
print('')
#LARGEST
print(f'Largest profit trade: {largest_profit}')
print(f'Largest loss trade: {largest_loss}')
print('')
#MAXWINS
print(f'Max Consecutive Wins: {max_element}')
print(f'Max Consecutive Wins value: {win_streaks_value[max_wins_ind]}')
print('')

#MAX
print(f'Max Consecutive loss: {max_low_element}')
print(f'Max Consecutive Wins value: {loss_streaks_value[max_loss_ind]}')
print('')
#MAXIMAL
print(f'Maximal consecutive profit (count): {max_consecutive_profit_count} {max_consecutive_profit_value:.2f}')
print(f'Maximal consecutive loss (count): {max_consecutive_loss_count} {max_consecutive_loss_value:.2f}')
print('')

print(f'Balance Drawdown Absolute: {drawdown_absolute:.2f}')
print(f'Balance Drawdown Maximal: {drawdown_maximal:.2f}')
print(f'Balance Drawdown Relative: {drawdown_relative:.2f}%')
print('')

#########
#########
#########

# Calculate totals for the required columns


# Generate timestamp for file names
timestamp = time.strftime("%Y%m%d_%H%M%S")
account_name = account_info[24].replace(" ", "_")
html_file_name = f"Acct_Rprt_{account_info[0]}_{extra_text}{timestamp}.html"
png_file_name = f"Balance_Graph_{account_info[0]}_{timestamp}.png"
document_title = f"Trading Report for {account_info[0]} - {extra_text} - {timestamp}"
pdf_file_path = os.path.join(output_directory, f"{html_file_name}.pdf")

# Function to generate HTML report
def generate_html(df):
       
    ############################################
    plt.figure(figsize=(10, 6))
    plt.plot(df['entry_time'], df['balance'], linestyle='-', color='b', linewidth=2)

    # Adding titles and labels
    plt.title(f'Balance over Time for {account_info[0]}', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Balance', fontsize=12)

    # Rotate x-axis labels for better visibility if needed
    plt.xticks(rotation=45)

    # Save the plot as a PNG file
    plt.savefig(png_file_name, bbox_inches='tight', dpi=300)

    #########################################

    # Generate the HTML dynamically
    html_rows = ""
    for _, row in df.iterrows():
        html_rows += f"""
        <tr>
            <td>{row['position_id']}</td>
            <td>{row['symbol']}</td>
            <td>{row['entry_time']}</td>
            <td>{row['closing_time']}</td>
            <td>{row['entry_price']}</td>
            <td>{row['closing_price']}</td>
            <td>{row['volume']}</td>
            <td>{row['type']}</td>
            <td>{row['magic']}</td>
            <td>{row['swap']}</td>
            <td>{row['commission']}</td>
            <td>{row['profit']}</td>
            <td>{row['balance']}</td>
            <td>{row['comment']}</td>
            <td>{row['trade_duration']}</td>
            <td>{row['max_profit']}</td>
            <td>{row['profit_time']}</td>
            <td>{row['profit_per_second']}</td>

        </tr>
        """
    # HTML structure
    # Assuming you already have your account_info and result_df data
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">

        <title>{document_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 10px;
                line-height: 1.4;
                background-color: #f9f9f9;
                color: #333;
            }}

            h1, h3 {{
                text-align: center;
                color: #444;
            }}

            h1 {{
                font-size: 1.8rem;
                margin-bottom: 8px;
            }}

            h3 {{
                font-size: 1.1rem;
                margin: 4px 0;
            }}

            .container {{
                max-width: 1000px;
                margin: 0 auto;
                padding: 15px;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                table-layout: fixed;
            }}

            th, td {{
                padding: 5px; /* Reduced padding for more space */
                text-align: right;
                border: 1px solid #ddd;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: normal; /* Allow wrapping inside cells */
                word-wrap: break-word; /* Break long words to fit inside cells */
            }}

            th {{
                background-color: #f4f4f4;
                font-weight: bold;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            tr:hover {{
                background-color: #f1f1f1;
            }}

            .summary-table td {{
                text-align: left;
            }}

            img {{
                display: block;
                margin: 15px auto;
                max-width: 40%;
                height: auto;
            }}

            @media screen and (max-width: 768px) {{
                table, th, td {{
                    font-size: 0.8rem; /* Further reduce font size on smaller screens */
                }}

                h1 {{
                    font-size: 1.4rem;
                }}

                h3 {{
                    font-size: 0.9rem;
                }}
            }}
        </style>



    </head>
    <body>
        <h1>Trading Report</h1>
        <h3>Account: {account_info[0]}</h3>
        <h3>Company: {account_info[27]}</h3>
        <h3>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>

        <table>
            <thead>
                <tr>
                    <th>Position ID</th>
                    <th>Symbol</th>
                    <th>Entry Time</th>
                    <th>Closing Time</th>
                    <th>Entry Price</th>
                    <th>Closing Price</th>
                    <th>Volume</th>
                    <th>Type</th>
                    <th>Magic</th>
                    <th>Swap (£)</th>
                    <th>Commission (£)</th>
                    <th>Profit (£)</th>
                    <th>Balance (£)</th>
                    <th>Comment</th>
                    <th>Trade Duration (mins)</th>
                    <th>Max Profit (£)</th>
                    <th>Time In Profit (mins)</th>
                    <th>Profit / S</th>
                </tr>
            </thead>
            <tbody>
                {html_rows}
            </tbody>
        </table>

        <table class="summary-table">
            <tr>
                <td colspan="2">Balance:</td>
                <td colspan="2">{account_info[10]}</td>
                <td colspan="2">Free Margin:</td>
                <td colspan="2">{account_info[15]}</td>
            </tr>
            <tr>
                <td colspan="2">Credit Facility:</td>
                <td colspan="2">{account_info[11]}</td>
                <td colspan="2">Margin:</td>
                <td colspan="2">{account_info[14]}</td>
            </tr>
            <tr>
                <td colspan="2">Floating P/L:</td>
                <td colspan="2">{account_info[12]}</td>
                <td colspan="2">Margin Level:</td>
                <td colspan="2">{account_info[16]}</td>
            </tr>
            <tr>
                <td colspan="2">Equity:</td>
                <td colspan="2">{account_info[13]}</td>
            </tr>
        </table>

        <img src={png_file_name} title="Balance graph" alt="Graph">

        <table class="summary-table">
            <tr>
                <td>Total Net Profit:</td>
                <td>{result_df[result_df['type'] != 2]['profit'].sum():.2f}</td>
            </tr>
            <tr>
                <td>Gross Profit:</td>
                <td>{gross_profit:.2f}</td>
            </tr>
            <tr>
                <td>Gross Loss:</td>
                <td>{gross_loss:.2f}</td>
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Total Trades:</td>
                <td>{len(result_df)}</td>

                <td>Long Trades Won:</td>
                <td>{long_wins}</td>
            </tr>
            <tr>
                <td>Total Won:</td>
                <td>{long_wins + short_wins}</td>

                <td>Long Trades Lost:</td>
                <td>{long_loss}</td>
            </tr>
            <tr>
                <td>Total Lost:</td>
                <td>{long_loss + short_loss}</td>

                <td>Short Trades Won:</td>
                <td>{short_wins}</td>
            </tr>
            <tr>
                <td>Win Rate:</td>
                <td>{win_rate:.2f}%</td>

                <td>Short Trades Lost:</td>
                <td>{short_loss}</td>
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Largest Profit Trade:</td>
                <td>{largest_profit:.2f}</td>

                <td>Largest Loss Trade:</td>
                <td>{largest_loss:.2f}</td>
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Profit Factor:</td>
                <td>{profit_factor:.2f}</td>
                
                <td>Annualised Sharpe Ratio:</td>
                <td>{annualised_sharpe_ratio:.2f}</td>
                
            </tr>
            <tr>
                <td>Recovery Factor:</td>
                <td>{recovery_factor:.2f}</td>
                <td>Daily Sharpe Ratio:</td>
                <td>{sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Expected Pay:</td>
                <td>{expected_payout:.2f}</td>
                
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Average RRR:</td>
                <td>1:{rrr:.2f}</td>

                <td>Average win streak:</td>
                <td>{avg_win_streak:.2f} ({avg_win_streak_value:.2f})</td>

            </tr>

            <tr>
                <td>Average profit trade:</td>
                <td>{average_profit:.2f}</td>

                <td>Average loss streak:</td>
                <td>{avg_loss_streak:.2f} ({avg_loss_streak_value:.2f})</td>
            </tr>


            <tr>
                <td>Average loss trade:</td>
                <td>{average_loss:.2f}</td>

                
            </tr>


        </table>

        <table class="summary-table">
            <tr>
                <td>Average Max Profit (Monetary):</td>
                <td>{average_max_profit:.2f}</td>

                <td>Average Trade Duration (Minutes):</td>
                <td>{(average_trade_duration / 60):.2f}</td>

            </tr>
            <tr>
                <td>Average Time in Profit (Minutes):</td>
                <td>{(average_time_in_profit / 60):.2f}</td>


                <td>Average Time in Drawdown (Minutes):</td>
                <td>{(average_time_in_drawdown / 60):.2f}</td>
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Max Consecutive Wins:</td>
                <td>{max_element} ({(win_streaks_value[max_wins_ind]):.2f})</td>

                <td>Maximal consecutive profit (count):</td>
                <td>{max_consecutive_profit_count} ({max_consecutive_profit_value:.2f})</td>
            </tr>

            <tr>
                <td>Max Consecutive loss:</td>
                <td>{max_low_element} ({(loss_streaks_value[max_loss_ind]):.2f})</td>

                <td>Maximal consecutive loss (count):</td>
                <td>{max_consecutive_loss_count} ({max_consecutive_loss_value:.2f})</td>
            </tr>
        </table>

        <table class="summary-table">
            <tr>
                <td>Balance Drawdown Absolute:</td>
                <td>{drawdown_absolute:.2f}</td>
            </tr>
            <tr>
                <td>Balance Drawdown Maximal:</td>
                <td>{drawdown_maximal:.2f}</td>
            </tr>
            <tr>
                <td>Balance Drawdown Relative:</td>
                <td>{drawdown_relative:.2f}%</td>
            </tr>

        </table>

    </body>
    </html>
    """

    return html_content


# Generate the HTML report
html_report = generate_html(new_df)

with open(html_file_name, "w", encoding="utf-8") as file:
    file.write(html_report)

#print(f"HTML report generated successfully: {html_file_name}")

# Save as PDF
#time.sleep(2)
path_to_wkhtmltopdf = r'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe'  # Use raw string or forward slashes
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

options = {
    'page-size': 'A3',
    'enable-local-file-access': ''  # Allow access to local files like images
}
pdfkit.from_file(html_file_name, pdf_file_path, options=options, configuration=config)

#time.sleep(5)
# Convert the HTML to a PDF
#pdf_file_name = "trading_report.pdf"
#HTML(string=html_report).write_pdf(pdf_file_name)

#print(f"PDF report generated successfully: {pdf_file_name}")
