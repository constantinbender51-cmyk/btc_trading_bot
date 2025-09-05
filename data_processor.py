
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

