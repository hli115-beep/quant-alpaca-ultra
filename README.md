# quant-alpaca-ultra
Momentum rotation with Bayesian tuning, walk-forward validation, and risk-managed execution on Alpaca (scheduler + live trading)
quant-alpaca/

├─ alpaca_live_momentum_ultra_v3.py   # main strategy (tuning + WFO + execution + risk)

├─ alpaca_ultra_scheduler_v3.py       # daemon/scheduler (heartbeat, timed rebalance)

Quick start
bash
Copy code
pip install yfinance pandas numpy scikit-optimize scikit-learn alpaca-trade-api

# Windows PowerShell (persist credentials)

setx ALPACA_KEY    "YOUR_KEY"

setx ALPACA_SECRET "YOUR_SECRET"

setx ALPACA_BASE   "https://paper-api.alpaca.markets"

Run once (weekly mode):

py -3.12 "alpaca_live_momentum_ultra_v3.py" --force --mode weekly


Keep it running (daemon):

py -3.12 "alpaca_ultra_scheduler_v3.py" --mode weekly
