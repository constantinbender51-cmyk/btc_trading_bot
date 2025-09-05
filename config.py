import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your-api-key-here')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
CSV_FILE_PATH = "btc_hourly_data.csv"
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.95  # 95% of capital per trade
MAX_OPEN_POSITIONS = 1
STOP_LOSS_BUFFER = 0.002  # 0.2% buffer for stop loss
