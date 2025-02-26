import subprocess
import time
from func_general_functions import *
from variables_general import *
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd

# Retry MetaTrader5 initialization
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

account_info = mt5.account_info()
if account_info is None:
    quit()

# Convert to pandas DataFrame
df = pd.DataFrame(list(position_deals), columns=position_deals[0]._asdict().keys())
df['time'] = pd.to_datetime(df['time'], unit='s')

# Group by position_id and calculate entry and closing prices
output_data = []
cumulative_balance = 0

for position_id in df['position_id'].unique():
    position_deals = df[df['position_id'] == position_id]
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

# Get unique values for symbols and third comments
unique_symbols = result_df['symbol'].unique()
third_comments = ['V4.1_T1_daily', 'V4.1_T2_daily', 'V4.1_T3_daily']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Function to run report_creator.py for each filter condition
def run_script(script_path, symbol, trade_type=None, comment=None, weekday=None):
    """
    Run the report_creator.py script with specified arguments.
    """
    args = f'--symbol "{symbol}"'
    if trade_type is not None:
        args += f' --trade_type {trade_type}'
    if comment:
        args += f' --comment "{comment}"'
    if weekday:
        args += f' --weekday "{weekday}"'  # Adding weekday filter

    command = f'start cmd.exe /c python "{script_path}" {args}'
    subprocess.Popen(command, shell=True)
    time.sleep(1)  # Add a delay to avoid overlapping processes

if __name__ == "__main__":
    # Iterate over symbols
    for symbol in unique_symbols:
        # Comprehensive report for the symbol (all trades)
        run_script('mass_report_creator.py', symbol)

        # Buys (trade_type=0) and Sells (trade_type=1)
        run_script('mass_report_creator.py', symbol, trade_type=0)  # Buys
        run_script('mass_report_creator.py', symbol, trade_type=1)  # Sells

        # Thirds (specific comments)
        for comment in third_comments:
            run_script('mass_report_creator.py', symbol, comment=comment)
            run_script('mass_report_creator.py', symbol, comment=comment, trade_type=0)
            run_script('mass_report_creator.py', symbol, comment=comment, trade_type=1)
            

        # Generate reports per weekday
        for weekday in weekdays:
            filtered_weekday_df = result_df[result_df['entry_time'].dt.strftime("%A") == weekday]
            if not filtered_weekday_df.empty:
                run_script('mass_report_creator.py', symbol, weekday=weekday)  # Run with weekday filter
                run_script('mass_report_creator.py', symbol, trade_type=0, weekday=weekday)  # Buys on weekday
                run_script('mass_report_creator.py', symbol, trade_type=1, weekday=weekday)  # Sells on weekday
                
                # Weekdays with third comments
                for comment in third_comments:
                    run_script('mass_report_creator.py', symbol, comment=comment, weekday=weekday)
                    run_script('mass_report_creator.py', symbol, comment=comment, trade_type=0, weekday=weekday)
                    run_script('mass_report_creator.py', symbol, comment=comment, trade_type=1, weekday=weekday)