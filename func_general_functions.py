import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from func_general_functions import *
import time
from variables_general import *


def retryable_initialize(max_retries, delay_seconds, terminal_path, current_pass, current_account, current_server):
    for attempt in range(1, max_retries + 1):
        if mt5.initialize(terminal_path):
            
            authorized=mt5.login(login = current_account, password=current_pass, server=current_server)
            
            if authorized:
                # display trading account data 'as is'
                print(f'Connected to {mt5.account_info()[0]}')
                
            else:
                print("failed to connect at account #{}, error code: {}".format(current_account, mt5.last_error()))
                if datetime.now(timezone.utc).hour == 9:
                    time.sleep(1860)
            return True  # If successful, exit the loop and return True
        else:
            print(f"Attempt {attempt} failed to initialize, error code: {mt5.last_error()}")
            #time.sleep(delay_seconds)  # Wait for the specified time before the next attempt

    return False        
    #send_notification('initialisation failed', f'{mt_account} failed to connect')
    #raise MaxRetriesExceeded(f"Max retries ({max_retries}) reached. Initialization failed.")
    
