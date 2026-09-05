"""
Shared config for the on-demand analysis strategies used by the stock chatbot.
Kept separate from nse-swing-dashboard/scanner/config.py since this is its
own standalone project, but mirrors the same tuning style.
"""

# ---- Universal exit rule (across all 4 strategies) ----
MAX_HOLD_DAYS = 21          # ~1 calendar month of trading days -- hard cap on any trade's holding period
RISK_REWARD_MULT = 2.0      # fallback target = entry + risk * this
STOP_BUFFER_PCT = 0.5       # extra cushion below/above a calculated stop

# ---- 20/50 EMA Crossover settings ----
EMA_FAST = 20
EMA_SLOW = 50
CROSSOVER_LOOKBACK_DAYS = 20     # bars used to find a recent swing low for the stop
CROSSOVER_MIN_AVG_VOLUME = 500_000

# ---- Volume Breakout settings ----
BREAKOUT_LOOKBACK_DAYS = 20      # N-day high the close must break above
BREAKOUT_VOLUME_MULT = 1.5       # breakout-day volume vs its 20-day average
BREAKOUT_MIN_AVG_VOLUME = 500_000
