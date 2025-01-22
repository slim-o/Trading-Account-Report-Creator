# Trading Report Generator

This script generates a detailed HTML report of trading metrics from historical trade data in MetaTrader 5 (MT5). It is designed to provide comprehensive insights into trading performance, including cumulative balances, key trading metrics, and drawdowns.

---

## Features

- **Key Metrics**:
  - Win rate, profit factor, Sharpe ratio, recovery factor, and expected payout.
  - Gross profit and loss calculations.
  - Drawdowns (absolute, maximal, and relative).
  
- **Dynamic HTML Report**:
  - Generates an HTML report summarising all trades.
  - Includes cumulative metrics and balances dynamically calculated.
  - Customisable and styled tables.

- **Filtering Options**:
  - Filter trades by magic number, type, and comments.

- **Extensibility**:
  - Easily modifiable for new metrics or custom report styles.

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or later
- Required Python libraries (install via `requirements.txt`):

  ```bash
  pip install -r requirements.txt
