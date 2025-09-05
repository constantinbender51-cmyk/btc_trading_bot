
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your-api-key-here')
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    # Trading Configuration
    INITIAL_CAPITAL = 10000
    POSITION_SIZE = 0.95  # 95% of capital per trade
    MAX_OPEN_POSITIONS = 1
    FEE_RATE = 0.001  # 0.1% trading fee
    
    # Data Configuration
    DATA_FILE = "btc_hourly_data.csv"
    LOOKBACK_CANDLES = 50
    SAVE_SIGNALS = True
    SIGNALS_FILE = "generated_signals.json"
    
    # Analysis Configuration
    QUARTERLY_ANALYSIS = True
    EXPORT_TRADES = True
    TRADES_FILE = "trade_history.csv"

