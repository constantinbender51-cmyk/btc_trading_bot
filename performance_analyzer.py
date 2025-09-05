# requirements.txt
"""
pandas==2.1.3
numpy==1.24.3
requests==2.31.0
python-dotenv==1.0.0
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1
"""

# config.py
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

# data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load and preprocess BTC hourly data"""
        try:
            self.data = pd.read_csv(self.file_path)
            
            # Standardize column names
            self.data.columns = [col.strip().lower() for col in self.data.columns]
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Validate columns
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Convert timestamp
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            # Add technical indicators
            self._add_technical_indicators()
            
            print(f"Loaded {len(self.data)} hourly candles from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _add_technical_indicators(self):
        """Add basic technical indicators"""
        # Simple Moving Averages
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['sma_50'] = self.data['close'].rolling(50).mean()
        
        # RSI
        self.data['rsi'] = self._calculate_rsi(self.data['close'], 14)
        
        # Bollinger Bands
        self.data['bb_upper'], self.data['bb_lower'] = self._calculate_bollinger_bands(
            self.data['close'], 20, 2
        )
        
        # Volume SMA
        self.data['volume_sma'] = self.data['volume'].rolling(20).mean()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def get_ohlc_chunk(self, end_idx, lookback=50):
        """Get OHLC data chunk for analysis"""
        start_idx = max(0, end_idx - lookback + 1)
        chunk = self.data.iloc[start_idx:end_idx+1].copy()
        
        ohlc_data = []
        for _, row in chunk.iterrows():
            ohlc_data.append({
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'rsi': float(row['rsi']) if pd.notna(row['rsi']) else None,
                'sma_20': float(row['sma_20']) if pd.notna(row['sma_20']) else None,
                'sma_50': float(row['sma_50']) if pd.notna(row['sma_50']) else None,
                'bb_upper': float(row['bb_upper']) if pd.notna(row['bb_upper']) else None,
                'bb_lower': float(row['bb_lower']) if pd.notna(row['bb_lower']) else None
            })
        
        return ohlc_data

# signal_generator.py
import requests
import json
import time
from typing import Dict, Any, Optional
from config import Config
import pandas as pd

class SignalGenerator:
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.api_url = Config.DEEPSEEK_API_URL
        self.request_count = 0
        self.error_count = 0
        
    def generate_signal(self, ohlc_data: list) -> Optional[Dict[str, Any]]:
        """Generate trading signal using DeepSeek API"""
        
        prompt = f"""Analyze the following BTC/USDT hourly OHLC data and generate a trading signal.
Respond with ONLY a JSON object containing: signal (BUY|SELL|HOLD), stop_price, target_price, 
confidence (0-100), and reason.

OHLC Data (most recent last):
{json.dumps(ohlc_data, indent=2)}

Important: Consider technical analysis, price action, volume patterns, and market structure.
Provide realistic stop and target prices based on support/resistance levels.
Return ONLY JSON, no other text."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency trader with expertise in technical analysis. Analyze the provided OHLC data and generate trading signals based on market conditions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            self.request_count += 1
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    signal_data = json.loads(content)
                    return self._validate_signal(signal_data, ohlc_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response: {content}")
                    self.error_count += 1
                    return None
            else:
                print(f"API Error: {response.status_code} - {response.text}")
           # performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List
from trade_simulator import Trade

class PerformanceAnalyzer:
    def __init__(self, trades: List[Trade]):
        self.trades = trades
        self.trades_df = self._create_trades_dataframe()
        
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from trades"""
        if not self.trades:
            return pd.DataFrame()
            
        data = []
        for trade in self.trades:
            data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'signal': trade.signal,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'confidence': trade.confidence,
                'status': trade.status,
                'reason': trade.reason
            })
        
        return pd.DataFrame(data)
    
    def quarterly_analysis(self) -> pd.DataFrame:
        """Analyze performance by quarter"""
        if self.trades_df.empty:
            return pd.DataFrame()
            
        df = self.trades_df.copy()
        df['entry_quarter'] = df['entry_time'].dt.to_period('Q')
        
        quarterly_stats = df.groupby('entry_quarter').agg({
            'pnl': ['sum', 'count', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)
        
        quarterly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'avg_return_pct']
        
        # Add win rate
        win_rate = df.groupby('entry_quarter').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(2)
        quarterly_stats['win_rate_pct'] = win_rate
        
        return quarterly_stats
    
    def plot_performance(self):
        """Create performance visualization plots"""
        if self.trades_df.empty:
            print("No trades to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Performance Analysis', fontsize=16)
        
        # 1. Cumulative P&L
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        axes[0, 0].plot(self.trades_df['exit_time'], cumulative_pnl)
        axes[0, 0].set_title('Cumulative P&L')
        axes[0, 0].set_ylabel('P&L ($)')
        axes[0, 0].grid(True)
        
        # 2. Trade Distribution
        axes[0, 1].hist(self.trades_df['pnl'], bins=30, alpha=0.7, color='blue')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Win/Loss by Signal
        signal_performance = self.trades_df.groupby('signal')['pnl'].agg(['sum', 'count'])
        signal_performance.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Performance by Signal Type')
        axes[1, 0].set_ylabel('P&L ($)')
        
        # 4. Monthly Performance
        monthly_pnl = self.trades_df.copy()
        monthly_pnl['month'] = monthly_pnl['exit_time'].dt.to_period('M')
        monthly_summary = monthly_pnl.groupby('month')['pnl'].sum()
        monthly_summary.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Monthly P&L')
        axes[1, 1].set_ylabel('P&L ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self) -> str:
        """Generate detailed performance report"""
        if self.trades_df.empty:
            return "No trades to analyze"
            
        metrics = self.calculate_detailed_metrics()
        
        report = f"""
TRADING PERFORMANCE REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2f}%
Profit Factor: {metrics['profit_factor']:.2f}
Total Return: {metrics['total_return']:.2f}%
Final Capital: ${metrics['final_capital']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2f}%

TRADE BREAKDOWN
----------------
Winning Trades: {metrics['winning_trades']}
Losing Trades: {metrics['losing_trades']}
Average Win: ${metrics['avg_win']:.2f} ({metrics['avg_win_pct']:.2f}%)
Average Loss: ${metrics['avg_loss']:.2f} ({metrics['avg_loss_pct']:.2f}%)
Largest Win: ${metrics['largest_win']:.2f}
Largest Loss: ${metrics['largest_loss']:.2f}

SIGNAL ANALYSIS
----------------
BUY Signals: {metrics['buy_signals']} (Win Rate: {metrics['buy_win_rate']:.2f}%)
SELL Signals: {metrics['sell_signals']} (Win Rate: {metrics['sell_win_rate']:.2f}%)

CONFIDENCE ANALYSIS
-------------------
Avg Confidence (All): {metrics['avg_confidence']:.2f}
Avg Confidence (Wins): {metrics['avg_confidence_wins']:.2f}
Avg Confidence (Losses): {metrics['avg_confidence_losses']:.2f}
"""
        
        return report
    
    def calculate_detailed_metrics(self) -> dict:
        """Calculate detailed performance metrics"""
        df = self.trades_df
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_win_pct = df[df['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = df[df['pnl'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        buy_signals = len(df[df['signal'] == 'BUY'])
        sell_signals = len(df[df['signal'] == 'SELL'])
        buy_win_rate = len(df[(df['signal'] == 'BUY') & (df['pnl'] > 0)]) / buy_signals * 100 if buy_signals > 0 else 0
        sell_win_rate = len(df[(df['signal'] == 'SELL') & (df['pnl'] > 0)]) / sell_signals * 100 if sell_signals > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss),
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': abs(avg_loss_pct),
            'largest_win': df['pnl'].max(),
            'largest_loss': abs(df['pnl'].min()),
            'profit_factor': profit_factor,
            'total_return': df['pnl'].sum() / (total_trades * 1000) * 100,  # Simplified
            'final_capital': 10000 + df['pnl'].sum(),  # Simplified
            'max_drawdown': 0,  # Would need capital tracking
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'avg_confidence': df['confidence'].mean(),
            'avg_confidence_wins': df[df['pnl'] > 0]['confidence'].mean() if winning_trades > 0 else 0,
            'avg_confidence_losses': df[df['pnl'] <= 0]['confidence'].mean() if losing_trades > 0 else 0
        }

