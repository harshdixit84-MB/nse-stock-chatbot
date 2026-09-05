"""
Shared config for the on-demand analysis strategies used by the stock chatbot.
Kept separate from nse-swing-dashboard/scanner/config.py since this is its
own standalone project, but mirrors the same tuning style.
"""

# ---- Universal exit rule (across all 4 strategies) ----
MAX_HOLD_DAYS = 21          # ~1 calendar month of trading days -- hard cap on any trade's holding period
RISK_REWARD_MULT = 2.0      # fallback target = entry + risk * this
STOP_BUFFER_PCT = 0.5       # extra cushion below/above a calculated stop

# ---- EMA Pullback settings ----
EMA_FAST = 20
EMA_SLOW = 50
SWING_LOOKBACK_DAYS = 20        # bars used to find the recent swing high
PULLBACK_MIN_PCT = 5.0          # minimum pullback from swing high
PULLBACK_MAX_PCT = 15.0         # maximum pullback from swing high
EMA_TOLERANCE_PCT = 2.0         # how close price must be to EMA20/EMA50
VOLUME_CONFIRM_MULT = 1.0       # reversal-day volume vs 20-day avg volume
MIN_AVG_VOLUME = 500_000        # 20-day average volume floor (liquidity filter)

# ---- 20/50 EMA Crossover settings ----
CROSSOVER_LOOKBACK_DAYS = 20     # bars used to find a recent swing low for the stop
CROSSOVER_MIN_AVG_VOLUME = 500_000

# ---- Volume Breakout settings ----
BREAKOUT_LOOKBACK_DAYS = 20      # N-day high the close must break above
BREAKOUT_VOLUME_MULT = 1.5       # breakout-day volume vs its 20-day average
BREAKOUT_MIN_AVG_VOLUME = 500_000
