
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
                self.error_count += 1
                return None
                
        except Exception as e:
            print(f"Error generating signal: {e}")
            self.error_count += 1
            return None
    
    def _validate_signal(self, signal_data: dict, ohlc_data: list) -> Optional[dict]:
        """Validate and clean signal data"""
        try:
            current_price = ohlc_data[-1]['close']
            
            # Validate signal type
            signal = signal_data.get('signal', 'HOLD').upper()
            if signal not in ['BUY', 'SELL', 'HOLD']:
                signal = 'HOLD'
            
            # Validate prices
            stop_price = float(signal_data.get('stop_price', current_price * 0.95))
            target_price = float(signal_data.get('target_price', current_price * 1.05))
            
            # Ensure stop and target are appropriate for signal
            if signal == 'BUY':
                stop_price = min(stop_price, current_price * 0.99)  # Stop below current
                target_price = max(target_price, current_price * 1.01)  # Target above
            elif signal == 'SELL':
                stop_price = max(stop_price, current_price * 1.01)  # Stop above current
                target_price = min(target_price, current_price * 0.99)  # Target below
            
            # Validate confidence
            confidence = max(0, min(100, float(signal_data.get('confidence', 50))))
            
            return {
                'signal': signal,
                'stop_price': round(stop_price, 2),
                'target_price': round(target_price, 2),
                'confidence': confidence,
                'reason': str(signal_data.get('reason', 'No reason provided'))[:200]
            }
            
        except Exception as e:
            print(f"Error validating signal: {e}")
            return None

