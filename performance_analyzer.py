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

