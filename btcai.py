# -*- coding: utf-8 -*-
#

# ===============================

BOT_TOKEN = "8204719284:AAGgttMEYnB6_e3060adppC5ITjfE3eC1-U"
CHAT_IDS = CHAT_IDS = [
    "5508995431",   # your id
    "1095699880"     # customer
]

def send_telegram(msg):

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        for chat_id in CHAT_IDS:

            payload = {
                "chat_id": chat_id,
                "text": msg
            }

            requests.post(url, data=payload)

    except Exception as e:
        print("Telegram Error:", e)


# alias function for compatibility
def send_telegram_alert(msg):
    send_telegram(msg)



import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# ===============================
# TRADE MANAGEMENT VARIABLES
# ===============================

active_trade = None
trade_log = pd.DataFrame(columns=[
    "entry_time",
    "exit_time",
    "entry",
    "exit",
    "pnl"
])

BASE_URL = "https://api.delta.exchange"

def get_data():

    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=180)).timestamp())

    url = f"{BASE_URL}/v2/history/candles"

    params = {
        "symbol":"BTCUSDT",
        "resolution":"5m",
        "start":start,
        "end":end
    }

    r = requests.get(url,params=params)
    data = r.json()["result"]

    df = pd.DataFrame(data,
        columns=["time","open","high","low","close","volume"]
    )

    df["time"] = pd.to_datetime(df["time"],unit="s")
    df.set_index("time",inplace=True)

    df = df.astype(float)

    return df

df = get_data()

df.tail()

df["ema20"] = df["close"].ewm(span=20).mean()
df["ema50"] = df["close"].ewm(span=50).mean()

df["rsi"] = 100 - (
    100/(1 + df["close"].pct_change().rolling(14).mean())
)

df["returns"] = df["close"].pct_change()

df["target"] = np.where(df["close"].shift(-1) > df["close"],1,0)

df.dropna(inplace=True)

features = ["ema20","ema50","rsi","returns"]

X = df[features]
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,shuffle=False
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6
)

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

print("Model Accuracy:",accuracy)

df["prediction"] = model.predict(X)

df["signal"] = 0

df.loc[df["prediction"]==1,"signal"] = 1
df.loc[df["prediction"]==0,"signal"] = -1

capital = 10000
risk = 0.01

position = None
entry_price = 0

trades = []
equity = []

for i in range(1,len(df)):

    price = df.close.iloc[i]
    signal = df.signal.iloc[i]

    if position is None and signal == 1:

        position = "long"
        entry_price = price

    elif position == "long" and signal == -1:

        pnl = price - entry_price

        capital += pnl

        trades.append({
            "entry":entry_price,
            "exit":price,
            "pnl":pnl,
            "time":df.index[i]
        })

        position = None

    equity.append(capital)

trade_log = pd.DataFrame(trades)

trade_log.head()

equity_series = pd.Series(equity)

returns = equity_series.pct_change().dropna()

total_return = (equity_series.iloc[-1]/equity_series.iloc[0]-1)*100

cagr = (
    (equity_series.iloc[-1]/equity_series.iloc[0])
    **(365/180)-1
)*100

max_dd = (
    (equity_series / equity_series.cummax()) - 1
).min()*100

sharpe = returns.mean()/returns.std()*np.sqrt(252)

print("Total Return %:",round(total_return,2))
print("CAGR %:",round(cagr,2))
print("Max Drawdown %:",round(max_dd,2))
print("Sharpe Ratio:",round(sharpe,2))
print("Total Trades:",len(trade_log))

trade_log["month"] = trade_log["time"].dt.to_period("M")

monthly_profit = trade_log.groupby("month")["pnl"].sum()

print(monthly_profit)

def create_features(df):

    # EMA indicators
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    # Returns
    df["returns"] = df["close"].pct_change()

    # RSI
    df["rsi"] = 100 - (100/(1 + df["returns"].rolling(14).mean()))

    # ATR calculation
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())

    df["tr"] = df[["high_low","high_close","low_close"]].max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    df.dropna(inplace=True)

    return df

trades_per_month = trade_log.groupby("month").size()

print(trades_per_month)

plt.figure(figsize=(12,6))

plt.plot(equity_series)

plt.title("Equity Curve")

plt.show()

import requests
import pandas as pd
import time

# =====================================
# GET MARKET CANDLE DATA
# =====================================

def get_live_data():

    try:

        # current time
        end_time = int(time.time())

        # last 200 minutes
        start_time = end_time - (200 * 60)

        url = f"https://api.delta.exchange/v2/history/candles?symbol=BTCUSDT&resolution=1m&start={start_time}&end={end_time}"

        r = requests.get(url)
        data = r.json()

        if "result" not in data:
            print("API Error:", data)
            return None

        candles = data["result"]

        if len(candles) == 0:
            print("No candles returned")
            return None

        df = pd.DataFrame(
            candles,
            columns=["time","open","high","low","close","volume"]
        )

        # ===============================
        # TIME CONVERSION
        # ===============================

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["time"] = df["time"].dt.tz_convert("Asia/Kolkata")

        # ===============================
        # NUMERIC
        # ===============================

        numeric_cols = ["open","high","low","close","volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        df.sort_values("time", inplace=True)

        df.set_index("time", inplace=True)

        return df

    except Exception as e:

        print("Data fetch error:", e)
        return None

def predict_market():

    global active_trade, trade_log

    df = get_live_data()

    if df is None:
        print("Skipping cycle...")
        return

    df = create_features(df)

    features = ["ema20","ema50","rsi","returns"]

    latest = df.iloc[-1]

    price = get_live_price()

    if price is None:
        print("Live price unavailable, skipping cycle...\n")
        return

    atr = latest["atr"]

    # =========================
    # CHECK ACTIVE TRADE
    # =========================

    if active_trade is not None:

        signal = active_trade["signal"]
        entry = active_trade["entry"]
        sl = active_trade["sl"]
        tp = active_trade["tp"]
        entry_time = active_trade["entry_time"]

        print("Time:", latest.name.strftime("%Y-%m-%d %H:%M:%S"))
        print("Current Price:", price)

        # ================= BUY =================

        if signal == "BUY":

            if price <= sl:

                pnl = price - entry

                print("STOP LOSS HIT ❌")

                msg = f"""
❌ STOP LOSS HIT

Signal: BUY
Entry: {entry}
Exit: {price}
PnL: {round(pnl,2)}

Entry Time: {entry_time}
Exit Time: {latest.name}
"""

                send_telegram(msg)

                trade_log.loc[len(trade_log)] = [
                    entry_time,
                    latest.name,
                    entry,
                    price,
                    pnl
                ]

                active_trade = None
                print("Waiting for next signal...\n")
                return


            elif price >= tp:

                pnl = price - entry

                print("TAKE PROFIT HIT ✅")

                msg = f"""
✅ TAKE PROFIT HIT

Signal: BUY
Entry: {entry}
Exit: {price}
PnL: {round(pnl,2)}

Entry Time: {entry_time}
Exit Time: {latest.name}
"""

                send_telegram(msg)

                trade_log.loc[len(trade_log)] = [
                    entry_time,
                    latest.name,
                    entry,
                    price,
                    pnl
                ]

                active_trade = None
                print("Waiting for next signal...\n")
                return


        # ================= SELL =================

        elif signal == "SELL":

            if price >= sl:

                pnl = entry - price

                print("STOP LOSS HIT ❌")

                msg = f"""
❌ STOP LOSS HIT

Signal: SELL
Entry: {entry}
Exit: {price}
PnL: {round(pnl,2)}

Entry Time: {entry_time}
Exit Time: {latest.name}
"""

                send_telegram(msg)

                trade_log.loc[len(trade_log)] = [
                    entry_time,
                    latest.name,
                    entry,
                    price,
                    pnl
                ]

                active_trade = None
                print("Waiting for next signal...\n")
                return


            elif price <= tp:

                pnl = entry - price

                print("TAKE PROFIT HIT ✅")

                msg = f"""
✅ TAKE PROFIT HIT

Signal: SELL
Entry: {entry}
Exit: {price}
PnL: {round(pnl,2)}

Entry Time: {entry_time}
Exit Time: {latest.name}
"""

                send_telegram(msg)

                trade_log.loc[len(trade_log)] = [
                    entry_time,
                    latest.name,
                    entry,
                    price,
                    pnl
                ]

                active_trade = None
                print("Waiting for next signal...\n")
                return

        print("Trade still running...\n")
        return


    # =========================
    # CREATE NEW TRADE SIGNAL
    # =========================

    X_live = pd.DataFrame([latest[features]], columns=features)

    prediction = model.predict(X_live)[0]

    signal = "BUY" if prediction == 1 else "SELL"

    RR = 2

    if signal == "BUY":
        sl = price - atr
        tp = price + (atr * RR)

    else:
        sl = price + atr
        tp = price - (atr * RR)

    active_trade = {
        "signal": signal,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "entry_time": latest.name
    }

    print("Time:", latest.name.strftime("%Y-%m-%d %H:%M:%S"))
    print("NEW TRADE 🚀")
    print("Signal:", signal)
    print("Entry Price:", round(price,2))
    print("Entry Time:", latest.name)
    print("Stop Loss:", round(sl,2))
    print("Take Profit:", round(tp,2))
    print()

    msg = f"""
🚀 NEW TRADE

Signal: {signal}
Entry Price: {round(price,2)}
Stop Loss: {round(sl,2)}
Take Profit: {round(tp,2)}

Entry Time: {latest.name}
"""

    send_telegram(msg)

# ===============================
# TIME CONVERSION (UTC → IST)
# ===============================

import pytz

IST = pytz.timezone("Asia/Kolkata")

def to_ist(ts):

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert(IST)

# ===============================
# DAILY PERFORMANCE REPORT
# ===============================

last_report_date = None

def send_daily_report():

    global trade_log, last_report_date

    if len(trade_log) == 0:
        return

    today = pd.Timestamp.now().date()

    # avoid sending multiple reports in same day
    if last_report_date == today:
        return

    # filter today's trades
    trades_today = trade_log[
        trade_log["exit_time"].dt.date == today
    ]

    if len(trades_today) == 0:
        return

    total_trades = len(trades_today)
    wins = len(trades_today[trades_today["pnl"] > 0])
    losses = len(trades_today[trades_today["pnl"] <= 0])

    win_rate = (wins / total_trades) * 100
    total_pnl = trades_today["pnl"].sum()

    avg_profit = trades_today["pnl"].mean()

    best_trade = trades_today["pnl"].max()
    worst_trade = trades_today["pnl"].min()

    msg = f"""
📊 DAILY BOT REPORT

Date: {today}

Total Trades: {total_trades}
Winning Trades: {wins}
Losing Trades: {losses}

Win Rate: {round(win_rate,2)}%

Total PnL: {round(total_pnl,2)}
Average Trade: {round(avg_profit,2)}

Best Trade: {round(best_trade,2)}
Worst Trade: {round(worst_trade,2)}
"""

    send_telegram(msg)

    last_report_date = today

import time

while True:

    try:

        predict_market()
        send_daily_report()

    except Exception as e:

        print("Loop error:", e)

    print("Waiting...\n")

    time.sleep(300)

send_telegram("✅ Bot test message working!")
