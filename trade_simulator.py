# trade_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from config import Config

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime = None
    signal: str = None
    entry_price: float = 0
    exit_price: float = 0
    stop_price: float = 0
    target_price: float = 0
    confidence: float = 0
    position_size: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    status: str = 'OPEN'  # OPEN, CLOSED, STOPPED, TARGET
    reason: str = ''

class TradeSimulator:
    def __init__(self, initial_capital: float = Config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
    def process_signal(self, timestamp: datetime, signal_data: dict, current_price: float):
        """Process trading signal and manage positions"""
        
        # Close existing positions if signal is opposite
        if signal_data['signal'] in ['BUY', 'SELL']:
            self._close_opposite_positions(signal_data['signal'], timestamp, current_price)
        
        # Open new position if no conflicting positions exist
        if signal_data['signal'] == 'BUY' and not self._has_long_position():
            self._open_long_position(timestamp, signal_data, current_price)
        elif signal_data['signal'] == 'SELL' and not self._has_short_position():
            self._open_short_position(timestamp, signal_data, current_price)
        
        # Check existing positions
        self._check_positions(timestamp, current_price)
        
    def _open_long_position(self, timestamp: datetime, signal_data: dict, current_price: float):
        """Open long position"""
        position_size = self.current_capital * Config.POSITION_SIZE
        shares = position_size / current_price
        
        trade = Trade(
            entry_time=timestamp,
            signal='BUY',
            entry_price=current_price,
            stop_price=signal_data['stop_price'],
            target_price=signal_data['target_price'],
            confidence=signal_data['confidence'],
            position_size=shares,
            reason=signal_data['reason']
        )
        
        self.positions.append(trade)
        print(f"LONG ENTRY: {timestamp} | Price: {current_price:.2f} | Shares: {shares:.4f}")
        
    def _open_short_position(self, timestamp: datetime, signal_data: dict, current_price: float):
        """Open short position"""
        position_size = self.current_capital * Config.POSITION_SIZE
        
        trade = Trade(
            entry_time=timestamp,
            signal='SELL',
            entry_price=current_price,
            stop_price=signal_data['stop_price'],
            target_price=signal_data['target_price'],
            confidence=signal_data['confidence'],
            position_size=position_size,
            reason=signal_data['reason']
        )
        
        self.positions.append(trade)
        print(f"SHORT ENTRY: {timestamp} | Price: {current_price:.2f} | Position: {position_size:.2f}")
        
    def _check_positions(self, timestamp: datetime, current_price: float):
        """Check and manage open positions"""
        for trade in self.positions[:]:  # Copy list to avoid modification during iteration
            if trade.signal == 'BUY':  # Long position
                if current_price <= trade.stop_price:
                    self._close_trade(trade, timestamp, current_price, 'STOPPED')
                elif current_price >= trade.target_price:
                    self._close_trade(trade, timestamp, current_price, 'TARGET')
            elif trade.signal == 'SELL':  # Short position
                if current_price >= trade.stop_price:
                    self._close_trade(trade, timestamp, current_price, 'STOPPED')
                elif current_price <= trade.target_price:
                    self._close_trade(trade, timestamp, current_price, 'TARGET')
                    
    def _close_trade(self, trade: Trade, timestamp: datetime, current_price: float, status: str):
        """Close a trade and calculate P&L"""
        trade.exit_time = timestamp
        trade.exit_price = current_price
        trade.status = status
        
        if trade.signal == 'BUY':
            trade.pnl = (current_price - trade.entry_price) * trade.position_size
            trade.pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
        else:  # SHORT
            trade.pnl = (trade.entry_price - current_price) * trade.position_size
            trade.pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
        
        # Apply fees
        fee = trade.position_size * current_price * Config.FEE_RATE * 2  # Entry + exit
        trade.pnl -= fee
        
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.positions.remove(trade)
        self.closed_trades.append(trade)
        
        print(f"TRADE CLOSED: {timestamp} | Signal: {trade.signal} | Status: {status} | "
              f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%) | Capital: ${self.current_capital:.2f}")
        
    def _close_opposite_positions(self, new_signal: str, timestamp: datetime, current_price: float):
        """Close positions opposite to new signal"""
        for trade in self.positions[:]:
            if (new_signal == 'BUY' and trade.signal == 'SELL') or \
               (new_signal == 'SELL' and trade.signal == 'BUY'):
                self._close_trade(trade, timestamp, current_price, 'CLOSED')
                
    def _has_long_position(self) -> bool:
        """Check if has open long position"""
        return any(trade.signal == 'BUY' for trade in self.positions)
    
    def _has_short_position(self) -> bool:
        """Check if has open short position"""
        return any(trade.signal == 'SELL' for trade in self.positions)
        
    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'signal': t.signal,
                'confidence': t.confidence,
                'status': t.status
            } for t in self.closed_trades
        ])
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = trades_df['pnl_pct']
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            'total_trades': len(self.closed_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf'),
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.current_capital
        }

