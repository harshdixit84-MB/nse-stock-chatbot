"""
Shared config for the on-demand analysis strategies used by the stock chatbot.
Kept separate from nse-swing-dashboard/scanner/config.py since this is its
own standalone project, but mirrors the same tuning style.
"""

# ---- Universal exit rule (across all 5 strategies) ----
MAX_HOLD_DAYS = 21          # ~1 calendar month of trading days -- hard cap on any trade's holding period
RISK_REWARD_MULT = 2.0      # fallback target = entry + risk * this

# STOP_BUFFER_PCT is superseded by the ATR-based buffer below (see
# core/risk.py). Left here only in case anything external still
# imports it -- no strategy module uses it anymore.
STOP_BUFFER_PCT = 0.5

# ---- ATR-based stop buffer (replaces the fixed % buffer above) ----
# Buffer = ATR_STOP_MULT * ATR(ATR_PERIOD), subtracted from each
# strategy's own structural stop level (swing low / EMA50 / support /
# pivot low). Scales the buffer to the stock's actual recent
# volatility instead of a flat percentage of price.
ATR_PERIOD = 14
ATR_STOP_MULT = 0.5

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

# ---- Price Action settings (pure candlestick + S/R, no EMA/RSI) ----
PA_SWING_LOOKBACK = 5            # bars each side for a fractal swing high/low
PA_SR_LOOKBACK_DAYS = 120        # how far back to look for support/resistance levels
PA_SR_TOUCH_TOLERANCE_PCT = 1.0  # % tolerance for grouping nearby touches into one level
PA_SR_MIN_TOUCHES = 2            # minimum touches for a level to count as real S/R
PA_NEAR_SUPPORT_PCT = 2.0        # how close price must be to the support level to qualify
PA_MIN_AVG_VOLUME = 500_000      # same liquidity floor as the other strategies
