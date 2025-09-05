# railway_app.py  (tiny health web process)
import os
from fastapi import FastAPI
from threading import Thread
import main

app = FastAPI(title="BTC-Backtester")

@app.get("/health")
def health():
    return {"status": "alive", "stage": os.getenv("RAILWAY_STAGE", "unknown")}

def run_backtest_once():
    if os.getenv("RUN_MODE") == "worker" or os.getenv("RUN_MODE") is None:
        main.main()

Thread(target=run_backtest_once, daemon=True).start()
