# README.md

# BTC Trading Bot Backtester

A comprehensive Python project for backtesting BTC trading strategies using AI-generated signals from the DeepSeek API.

---

## Features

- **AI-Powered Signals**: Leverages DeepSeek API to analyze 50 hours of BTC OHLC data and generate actionable trading signals
- **Complete Backtesting Engine**: Simulates trades with realistic stop-loss and take-profit levels
- **Performance Analytics**:
  - Quarterly returns breakdown
  - Win/loss ratios and P&L tracking
  - Maximum drawdown and Sharpe ratio
  - Signal confidence vs. accuracy analysis
- **Technical Indicators**: RSI, SMA, Bollinger Bands automatically included in prompt
- **Intelligent Caching**: Saves generated signals to JSON to avoid redundant API calls and reduce cost
- **Rich Visualizations**: Equity curve, trade distribution, monthly/quarterly heatmaps
- **Export Options**: CSV exports of all trades, human-readable performance reports

---

## Quick Start

1. Clone / download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
