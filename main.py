# main.py
import pandas as pd
import json
import time
from datetime import datetime
from tqdm import tqdm
import os
from typing import Optional

from config import Config
from data_processor import DataProcessor
from signal_generator import SignalGenerator
from trade_simulator import TradeSimulator
from performance_analyzer import PerformanceAnalyzer

class BTCBacktester:
    def __init__(self):
        self.data_processor = DataProcessor(Config.DATA_FILE)
        self.signal_generator = SignalGenerator()
        self.simulator = TradeSimulator()
        self.signals_cache = {}

    def run_backtest(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        """Run the complete backtest"""
        print("Loading BTC data...")
        if not self.data_processor.load_data():
            return

        data = self.data_processor.data
        total_candles = len(data)

        start_idx = start_idx or Config.LOOKBACK_CANDLES
        end_idx   = end_idx or total_candles - 1

        print(f"Running backtest from candle {start_idx} to {end_idx}...")
        print(f"Total candles to process: {end_idx - start_idx + 1}")

        self._load_signals_cache()

        for i in tqdm(range(start_idx, end_idx + 1), desc="Processing candles"):
            current_data = data.iloc[i]
            timestamp    = current_data['timestamp']

            ohlc_data = self.data_processor.get_ohlc_chunk(i, Config.LOOKBACK_CANDLES)

            signal_key = f"{timestamp}_{i}"
            if signal_key in self.signals_cache:
                signal_data = self.signals_cache[signal_key]
            else:
                signal_data = self.signal_generator.generate_signal(ohlc_data)
                if signal_data and Config.SAVE_SIGNALS:
                    self.signals_cache[signal_key] = signal_data

            if signal_data:
                current_price = current_data['close']
                self.simulator.process_signal(timestamp, signal_data, current_price)

            time.sleep(0.1)

            if i % 100 == 0 and Config.SAVE_SIGNALS:
                self._save_signals_cache()

        final_data = data.iloc[end_idx]
        self.simulator._close_all_positions(final_data['timestamp'], final_data['close'])
        self._save_signals_cache()
        self._generate_performance_report()

    def _load_signals_cache(self):
        if os.path.exists(Config.SIGNALS_FILE):
            try:
                with open(Config.SIGNALS_FILE, 'r') as f:
                    self.signals_cache = json.load(f)
                print(f"Loaded {len(self.signals_cache)} cached signals")
            except Exception as e:
                print(f"Error loading signals cache: {e}")
                self.signals_cache = {}

    def _save_signals_cache(self):
        if Config.SAVE_SIGNALS:
            try:
                with open(Config.SIGNALS_FILE, 'w') as f:
                    json.dump(self.signals_cache, f, indent=2)
            except Exception as e:
                print(f"Error saving signals cache: {e}")

    def _generate_performance_report(self):
        metrics = self.simulator.get_performance_metrics()
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Final Capital: ${metrics['final_capital']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        analyzer = PerformanceAnalyzer(self.simulator.closed_trades)
        if Config.QUARTERLY_ANALYSIS:
            quarterly_stats = analyzer.quarterly_analysis()
            print("\nQUARTERLY PERFORMANCE:")
            print(quarterly_stats)

        report = analyzer.generate_report()
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {report_file}")

        try:
            analyzer.plot_performance()
        except Exception as e:
            print(f"Error creating plots: {e}")

        if Config.EXPORT_TRADES and self.simulator.closed_trades:
            trades_df   = analyzer.trades_df
            trades_file = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"Trade history exported to: {trades_file}")


def main():
    print("BTC Trading Bot Backtester")
    print("="*50)

    if Config.DEEPSEEK_API_KEY == 'your-api-key-here':
        print("ERROR: Please set your DeepSeek API key in the .env file")
        print("Create a .env file with: DEEPSEEK_API_KEY=your_actual_api_key")
        return

    backtester = BTCBacktester()
    try:
        backtester.run_backtest()
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
    except Exception as e:
        print(f"Error during backtest: {e}")


if __name__ == "__main__":
    main()
  
