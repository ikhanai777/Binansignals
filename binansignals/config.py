"""Static configuration: the 12 pairs, cost model and strategy defaults."""

# The 12 requested pairs. MATICUSDT was delisted from Binance USD-M futures
# after the MATIC -> POL token migration (Sept 2024); POLUSDT is the live
# successor contract, so it is used as the tradable symbol for "Polygon".
PAIRS = [
    {"pair": "BTCUSDT", "symbol": "BTCUSDT", "name": "Bitcoin", "keywords": ["bitcoin", "btc"]},
    {"pair": "ETHUSDT", "symbol": "ETHUSDT", "name": "Ethereum", "keywords": ["ethereum", "ether", "eth"]},
    {"pair": "XRPUSDT", "symbol": "XRPUSDT", "name": "XRP", "keywords": ["xrp", "ripple"]},
    {"pair": "LTCUSDT", "symbol": "LTCUSDT", "name": "Litecoin", "keywords": ["litecoin", "ltc"]},
    {"pair": "DOGEUSDT", "symbol": "DOGEUSDT", "name": "Dogecoin", "keywords": ["dogecoin", "doge"]},
    {"pair": "DOTUSDT", "symbol": "DOTUSDT", "name": "Polkadot", "keywords": ["polkadot", "dot"]},
    {"pair": "AVAXUSDT", "symbol": "AVAXUSDT", "name": "Avalanche", "keywords": ["avalanche", "avax"]},
    {"pair": "BNBUSDT", "symbol": "BNBUSDT", "name": "BNB", "keywords": ["bnb", "bnb chain"]},
    {"pair": "SOLUSDT", "symbol": "SOLUSDT", "name": "Solana", "keywords": ["solana", "sol"]},
    {"pair": "MATICUSDT", "symbol": "POLUSDT", "name": "Polygon (POL, ex-MATIC)", "keywords": ["polygon", "matic", "pol"]},
    {"pair": "LINKUSDT", "symbol": "LINKUSDT", "name": "Chainlink", "keywords": ["chainlink", "link"]},
    {"pair": "ADAUSDT", "symbol": "ADAUSDT", "name": "Cardano", "keywords": ["cardano", "ada"]},
]

# Binance USD-M futures REST hosts, tried in order. fapi.binance.com is the
# official host; www.binance.com/fapi serves the same public endpoints and is
# reachable from some networks where fapi.binance.com is geo-blocked.
FAPI_HOSTS = ["https://fapi.binance.com", "https://www.binance.com"]

# Cost model (Binance USD-M VIP0 regular fees).
TAKER_FEE = 0.0005   # 0.05% market / stop orders
MAKER_FEE = 0.0002   # 0.02% resting limit orders (take-profits)
SLIPPAGE = 0.0002    # 0.02% assumed slippage on every market / stop fill

# Account defaults for position sizing in live signals.
RISK_PER_TRADE = 0.01   # risk 1% of equity per signal
MAX_LEVERAGE = 10

# Walk-forward backtest windows (in days).
WF_TRAIN_DAYS = 180
WF_TEST_DAYS = 60
BACKTEST_DAYS = 1095    # ~3 years of 1h history (less if the contract is younger)

# Minimum out-of-sample quality before a pair's signals are marked tradable.
MIN_OOS_TRADES = 25
MIN_OOS_PROFIT_FACTOR = 1.15
