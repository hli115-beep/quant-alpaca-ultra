# -*- coding: utf-8 -*-
# 省略：文件头注释同上

import os, sys, json, time, subprocess, math
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo

from alpaca_trade_api.rest import REST
import yfinance as yf

# 修改为你的脚本路径
SCRIPT_PATH = r"C:\Users\joey2\OneDrive - University of Rochester\Desktop\量化\alpaca_live_momentum_ultra_v2.py"
LOG_PATH    = os.path.join(os.path.dirname(SCRIPT_PATH), "ultra_daemon.log")
STATE_FILE  = os.path.join(os.path.dirname(SCRIPT_PATH), "daemon_state.json")

TZ_NY = ZoneInfo("America/New_York")
WEEKLY_REBAL_TIME = (15, 45)

GUARD_CHECK_SEC     = 30
CIRCUIT_DD_INTRADAY = 0.05
CIRCUIT_DD_TOTAL    = 0.20
TRAIL_STOP          = 0.08
DAY_DROP_STOP       = 0.06
OPEN_GAP_PROTECT    = 0.03
VIX_ABS_LEVEL       = 28.0
VIX_SPIKE_PCT       = 0.20
DELEVER_SCALE       = 0.5

def log(msg):
    ts = datetime.now(TZ_NY).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f: f.write(line + "\n")
    except Exception: pass

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"last_weekly_run": None,"intraday_peak_equity": None,"symbol_peaks": {},"block_new_buys_today": False}

def save_state(s):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"[WARN] save_state failed: {e}")

def env(name:str)->str:
    v=os.getenv(name)
    if not v: raise RuntimeError(f"环境变量 {name} 未设置")
    return v

def new_rest():
    return REST(key_id=env("ALPACA_KEY"), secret_key=env("ALPACA_SECRET"),
                base_url=os.getenv("ALPACA_BASE","https://paper-api.alpaca.markets"), api_version="v2")

def is_trading_day(api: REST, d: date) -> bool:
    return len(api.get_calendar(start=d.isoformat(), end=d.isoformat())) > 0

def last_trading_day_of_week(api: REST, d: date) -> bool:
    monday = d - timedelta(days=d.weekday()); sunday = monday + timedelta(days=6)
    cals = api.get_calendar(start=monday.isoformat(), end=sunday.isoformat())
    return bool(cals) and cals[-1].date == d

def weekly_rebal_dt(api: REST, d: date):
    if not is_trading_day(api, d) or not last_trading_day_of_week(api, d): return None
    return datetime(d.year,d.month,d.day,WEEKLY_REBAL_TIME[0],WEEKLY_REBAL_TIME[1],tzinfo=TZ_NY)

def latest_price_alpaca(api: REST, symbol: str):
    try: return float(api.get_latest_trade(symbol).price)
    except Exception: return None

def latest_price(symbol: str, api: REST):
    p = latest_price_alpaca(api, symbol)
    if p is not None: return p
    try:
        df = yf.download(symbol, period="1d", interval="1m", auto_adjust=True, progress=False)
        if df is not None and not df.empty: return float(df["Close"].iloc[-1])
    except Exception: pass
    return None

def get_prev_close(symbol: str):
    try:
        df = yf.download(symbol, period="5d", interval="1d", auto_adjust=True, progress=False)
        if df is None or len(df)<2: return None
        return float(df["Close"].iloc[-2])
    except Exception: return None

def market_close_to_limit(api: REST, symbol: str, qty: int, ref_px: float, eps: float = 0.002):
    if qty==0: return
    side = "buy" if qty>0 else "sell"
    limit = ref_px*(1+eps if qty>0 else 1-eps)
    return api.submit_order(symbol=symbol, qty=abs(qty), side=side, type="limit",
                            time_in_force="day", limit_price=round(float(limit),2))

def liquidate_symbol(api: REST, symbol: str):
    try: api.cancel_orders()
    except Exception: pass
    try:
        api.close_position(symbol); log(f"[RISK] 清仓 {symbol}")
    except Exception as e:
        log(f"[WARN] 清仓失败 {symbol}: {e}")

def scale_down_positions(api: REST, scale: float):
    try:
        for p in api.list_positions():
            qty = float(p.qty)
            if qty==0: continue
            target = qty*scale; delta = int(round(target - qty))
            if delta==0: continue
            px = latest_price(p.symbol, api) or float(p.avg_entry_price)
            try:
                market_close_to_limit(api, p.symbol, delta, px)
                log(f"[RISK] 降权 {p.symbol} 目标比例 {scale:.2f} 提交:{delta}")
            except Exception as e:
                log(f"[WARN] 降权失败 {p.symbol}: {e}")
    except Exception as e:
        log(f"[WARN] scale_down_positions 异常: {e}")

def portfolio_value(api: REST) -> float:
    try:
        acc = api.get_account()
        cash = float(acc.cash)
        pv = float(acc.portfolio_value) if hasattr(acc,"portfolio_value") else cash
        if pv>0: return pv
    except Exception: pass
    try:
        cash = float(api.get_account().cash); tot = cash
        for p in api.list_positions():
            pr = latest_price(p.symbol, api) or float(p.avg_entry_price)
            tot += float(p.qty) * pr
        return tot
    except Exception: return 0.0

def intraday_guardian(api: REST, state: dict):
    today = datetime.now(TZ_NY).date()
    start = datetime(today.year,today.month,today.day,9,30,tzinfo=TZ_NY)
    end   = datetime(today.year,today.month,today.day,16,0,tzinfo=TZ_NY)

    if start <= datetime.now(TZ_NY) <= start + timedelta(minutes=5):
        pc = get_prev_close("SPY"); last = latest_price("SPY", api)
        if pc and last:
            gap = (last/pc - 1.0)
            if gap <= -OPEN_GAP_PROTECT:
                state["block_new_buys_today"] = True; save_state(state)
                log(f"[OPEN] SPY 低开 {gap:.2%} → 今日禁止新增买入")

    intraday_peak = state.get("intraday_peak_equity")
    hist_peak = state.get("hist_peak_equity")

    eq_path = os.path.join(os.path.dirname(SCRIPT_PATH), "equity_weekly.csv")
    if hist_peak is None:
        try:
            import pandas as pd
            eq = pd.read_csv(eq_path)
            if not eq.empty:
                hist_peak = float((eq["strategy_eq"]).cummax().iloc[-1])
                state["hist_peak_equity"] = hist_peak
        except Exception:
            hist_peak = None

    while start <= datetime.now(TZ_NY) <= end:
        try:
            pv = portfolio_value(api)
            if pv <= 0:
                time.sleep(GUARD_CHECK_SEC); continue

            if intraday_peak is None or pv > intraday_peak:
                intraday_peak = pv; state["intraday_peak_equity"] = intraday_peak; save_state(state)
            if hist_peak is None or pv > hist_peak:
                hist_peak = pv; state["hist_peak_equity"] = hist_peak; save_state(state)

            dd_intraday = pv/intraday_peak - 1.0 if intraday_peak else 0.0
            dd_total    = pv/hist_peak - 1.0 if hist_peak else 0.0

            try:
                vix = yf.download("^VIX", period="5d", interval="1m", progress=False)
                vix_last = float(vix["Close"].iloc[-1])
                vix_prev = float(vix["Close"].iloc[-390]) if len(vix)>390 else float(vix["Close"].iloc[0])
            except Exception:
                vix_last=vix_prev=None

            if dd_intraday <= -CIRCUIT_DD_INTRADAY:
                log(f"[CB] 当日回撤 {dd_intraday:.2%} 触发 → 降权 {DELEVER_SCALE:.0%}")
                scale_down_positions(api, DELEVER_SCALE)
                intraday_peak = pv
            elif dd_total <= -CIRCUIT_DD_TOTAL:
                log(f"[CB] 历史回撤 {dd_total:.2%} 触发 → 清仓")
                try: api.close_all_positions()
                except Exception as e: log(f"[WARN] close_all_positions 失败: {e}")
                intraday_peak = pv

            sym_peaks = state.get("symbol_peaks", {}); new_peaks = {}
            for p in api.list_positions():
                sym = p.symbol
                pr  = latest_price(sym, api) or float(p.avg_entry_price)
                pk  = max(float(sym_peaks.get(sym, pr)), pr)
                new_peaks[sym] = pk
                if pr/pk - 1.0 <= -TRAIL_STOP:
                    liquidate_symbol(api, sym)
                else:
                    pc = get_prev_close(sym)
                    if pc and (pr/pc - 1.0) <= -DAY_DROP_STOP:
                        liquidate_symbol(api, sym)
            state["symbol_peaks"] = new_peaks; save_state(state)

            if vix_last:
                spike = (vix_last/vix_prev - 1.0) if (vix_prev and vix_prev>0) else 0.0
                if vix_last >= VIX_ABS_LEVEL or spike >= VIX_SPIKE_PCT:
                    log(f"[VIX] {vix_last:.2f} / spike {spike:.2%} → 降权 {DELEVER_SCALE:.0%}")
                    scale_down_positions(api, DELEVER_SCALE)

            time.sleep(GUARD_CHECK_SEC)
        except Exception as e:
            log(f"[ERROR] 守护循环异常: {e}")
            time.sleep(GUARD_CHECK_SEC)

def run_rebalance_if_due(api: REST, state: dict):
    today = datetime.now(TZ_NY).date()
    dt = weekly_rebal_dt(api, today)
    if dt is None: return
    if datetime.now(TZ_NY) >= dt and state.get("last_weekly_run") != today.isoformat():
        envv = os.environ.copy()
        if state.get("block_new_buys_today", False):
            envv["ULTRA_BLOCK_BUYS"]="1"; log("[OPEN] 今日禁买标志已传递给主脚本")
        cmd = [sys.executable, "-u", SCRIPT_PATH, "--mode", "weekly", "--tag", "weekly"]
        log(f"[RUN] WEEKLY_REBAL → {cmd}")
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", env=envv)
        if cp.stdout:
            for line in cp.stdout.splitlines(): log("[OUT] " + line)
        if cp.stderr:
            for line in cp.stderr.splitlines(): log("[ERR] " + line)
        log(f"[DONE] returncode={cp.returncode}")
        state["last_weekly_run"] = today.isoformat(); save_state(state)

def main_loop():
    state = load_state()
    api = new_rest()
    log("== Ultra Scheduler v2 启动（周频 + 盘中风控）==")
    log(f"使用脚本：{SCRIPT_PATH}")
    log(f"日志：{LOG_PATH}")

    while True:
        now = datetime.now(TZ_NY)
        try:
            if not is_trading_day(api, now.date()):
                if state.get("block_new_buys_today", False):
                    state["block_new_buys_today"] = False; save_state(state)
                start = now.date() + timedelta(days=1)
                end = start + timedelta(days=10)
                cals = api.get_calendar(start=start.isoformat(), end=end.isoformat())
                if cals:
                    nxt = cals[0].date
                    wake = datetime(nxt.year,nxt.month,nxt.day,9,0,tzinfo=TZ_NY)
                    secs = (wake-now).total_seconds()
                    log(f"[SLEEP] 非交易日，睡到下个交易日 09:00（{int(max(0,secs))} 秒）")
                    time.sleep(min(max(0,secs), 3600))
                else:
                    time.sleep(3600)
                continue

            if now.time() >= datetime(now.year,now.month,now.day,9,30,tzinfo=TZ_NY).time() and \
               now.time() <= datetime(now.year,now.month,now.day,16,0,tzinfo=TZ_NY).time():
                intraday_guardian(api, state)

            run_rebalance_if_due(api, state)

            tomorrow = now.date() + timedelta(days=1)
            wake = datetime(tomorrow.year,tomorrow.month,tomorrow.day,9,20,tzinfo=TZ_NY)
            secs = (wake-now).total_seconds()
            log(f"[SLEEP] 今日流程结束，睡到明日 09:20（{int(max(0,secs))} 秒）")
            time.sleep(min(max(0,secs), 3600))

        except KeyboardInterrupt:
            log("== 捕获 Ctrl+C，退出 =="); break
        except Exception as e:
            log(f"[ERROR] 主循环异常：{e}")
            time.sleep(30)

if __name__ == "__main__":
    main_loop()
