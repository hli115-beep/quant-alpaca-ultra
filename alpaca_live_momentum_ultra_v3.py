# -*- coding: utf-8 -*-
# 省略：文件头注释同上

import os, json, csv, time, math, warnings, argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

UNIVERSE   = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","QQQ","SPY"]
START_DATE = "2015-01-01"
END_DATE   = None
INIT_CASH  = 100_000.0

PARAM_BOUNDS = {"LOOKBACK_M": (6, 18), "TOP_N": (3, 6), "SMA_WIN": (100, 220)}
USE_BAYES_OPT = True
N_INIT, N_TRIALS, PATIENCE = 10, 40, 8

RETRAIN_EVERY_M = 1
MIN_TRAIN_MONTHS = 36

COMMISSION, SLIPPAGE = 0.0003, 0.0005
ONE_WAY = COMMISSION + SLIPPAGE

MAX_WEIGHT, TARGET_VOL_ANN = 0.30, 0.20
MDD_THRESHOLD, MDD_ACTION, DELEVER_TO = -0.25, "delever", 0.5

W_SHARPE, W_CALMAR, W_ALPHA = 0.5, 0.3, 0.2

# ——关键：市价化限价参数＋轮询
LIMIT_EPS = 0.002   # 0.2% “可交易限价”缓冲
POLL_SEC  = 3

# 再平衡与阈值
REBALANCE_MODE = "weekly"
DEVIATION_THRESHOLD = 0.02

def parse_args_mode_tag():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", choices=["daily","weekly","monthly"], default=None)
    p.add_argument("--tag",  type=str, default=None)
    a,_ = p.parse_known_args()
    return a.mode, a.tag

mode_cli, tag_cli = parse_args_mode_tag()
REBALANCE_MODE = mode_cli or REBALANCE_MODE
TAG = (tag_cli or REBALANCE_MODE or "default").lower()

STATE_FILE  = f"state_{TAG}.json"
EQUITY_FILE = f"equity_{TAG}.csv"
TRADES_FILE = f"trades_{TAG}.csv"
PARAMS_FILE = f"params_{TAG}.json"

def _json_default(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))

def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_series().resample("ME").last().dropna().index

def last_trade_before(prices: pd.DataFrame, dt) -> pd.Series:
    return prices.loc[:dt].iloc[-1]

def max_drawdown(eq: pd.Series) -> float:
    roll = eq.cummax(); dd = eq/roll - 1.0
    return float(dd.min()) if len(dd) else 0.0

def perf_from_monthly(ret_m: pd.Series, bench_m: Optional[pd.Series]=None) -> Dict[str,float]:
    ret_m = ret_m.dropna()
    if ret_m.empty:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, calmar=np.nan, alpha=np.nan, te=np.nan, ir=np.nan)
    eq = (1+ret_m).cumprod(); n = len(ret_m)
    ann = eq.iloc[-1]**(12/n) - 1
    vol = ret_m.std()*math.sqrt(12)
    sharpe = ann/vol if vol>0 else np.nan
    mdd = max_drawdown(eq)
    calmar = ann/abs(mdd) if mdd<0 else np.nan
    alpha=te=ir=np.nan
    if bench_m is not None and not bench_m.empty:
        bench_m = bench_m.reindex_like(ret_m).dropna()
        common = ret_m.index.intersection(bench_m.index)
        if len(common)>6:
            ex = ret_m.loc[common] - bench_m.loc[common]
            alpha = (1+ex).prod()**(12/len(ex)) - 1
            te = ex.std()*math.sqrt(12)
            ir = alpha/te if te>0 else np.nan
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar, alpha=alpha, te=te, ir=ir)

def score_obj(m: Dict[str,float]) -> float:
    val = 0.0
    if not math.isnan(m["sharpe"]): val += W_SHARPE*m["sharpe"]
    if not math.isnan(m["calmar"]): val += W_CALMAR*m["calmar"]
    if not math.isnan(m["alpha"]):  val += W_ALPHA*m["alpha"]
    return float(val)

def download_prices(tickers, start, end=None):
    end = end or (pd.Timestamp.utcnow()+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    series_list = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            print(f"[WARN] {t} 无数据，跳过"); continue
        s = df["Close"].copy(); s.name = t
        series_list.append(s)
    if not series_list:
        raise RuntimeError("未获取到任何标的的数据")
    prices = pd.concat(series_list, axis=1).sort_index().ffill()
    prices.index = pd.to_datetime(prices.index)
    return prices

def compute_factors(prices_daily: pd.DataFrame, lookback_m:int, sma_win:int):
    monthly = prices_daily.resample("ME").last()
    mom = monthly.pct_change(lookback_m) - monthly.pct_change(1)
    sma = prices_daily.rolling(sma_win).mean().resample("ME").last()
    trend = (monthly/sma) - 1.0
    vol_m = monthly.pct_change().rolling(lookback_m).std()
    lowvol = -vol_m
    def z(df): return (df - df.mean(axis=1, skipna=True))/df.std(axis=1, ddof=0, skipna=True)
    f = {"momentum": z(mom), "trend": z(trend), "lowvol": z(lowvol)}
    for k in f: f[k] = f[k].reindex(monthly.index)
    return monthly, f

def learn_asset_scores(factors, monthly_prices, train_end):
    ret_m = monthly_prices.pct_change()
    idx = monthly_prices.index
    tickers = monthly_prices.columns
    X_rows, y_rows = [], []
    names = list(factors.keys())
    for i in range(1, len(idx)):
        t  = idx[i]; tp = idx[i-1]
        if t > train_end: break
        for sym in tickers:
            row=[factors[name].loc[tp, sym] for name in names]
            X_rows.append(row); y_rows.append(ret_m.loc[t, sym])
    X = np.nan_to_num(np.array(X_rows)); y = np.nan_to_num(np.array(y_rows))
    if len(X)==0: return pd.Series(0.0, index=tickers)
    model = Ridge(alpha=1.0, fit_intercept=True); model.fit(X, y)
    tp = idx[-2]
    Xp = np.nan_to_num(np.array([[factors[name].loc[tp, sym] for name in names] for sym in tickers]))
    pred = model.predict(Xp)
    return pd.Series(pred, index=tickers).replace([np.inf,-np.inf], 0).fillna(0.0)

def minvar_weights(rets_hist: pd.DataFrame, clip=MAX_WEIGHT) -> pd.Series:
    cols = rets_hist.columns
    if rets_hist.dropna().empty: return pd.Series(1.0/len(cols), index=cols)
    cov = rets_hist.cov().values; n = cov.shape[0]
    try:
        inv = np.linalg.pinv(cov); ones = np.ones((n,1)); raw = inv.dot(ones)
        w = raw/float(ones.T.dot(inv).dot(ones)); w = pd.Series(w.flatten(), index=cols)
    except Exception:
        vol = rets_hist.std().replace(0,np.nan)
        w = (1/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
        w = w/w.sum() if w.sum()>0 else pd.Series(1.0/len(cols), index=cols)
    w = w.clip(0, clip); w = w/(w.sum() if w.sum()>0 else 1.0)
    return w

def scale_to_target_vol(w: pd.Series, rets_hist: pd.DataFrame, target_vol=TARGET_VOL_ANN) -> pd.Series:
    if not target_vol or target_vol<=0 or rets_hist.dropna().empty: return w
    vol_est = (rets_hist.dot(w)).std()*math.sqrt(12)
    if not vol_est or vol_est<=0: return w
    scale = min(2.0, max(0.2, target_vol/vol_est))
    w = (w*scale).clip(0, MAX_WEIGHT); w = w/(w.sum() if w.sum()>0 else 1.0)
    return w

def turnover_cost(prev_w: Optional[pd.Series], target_w: pd.Series, ret_m_row: pd.Series, one_way=ONE_WAY):
    if prev_w is None or prev_w.sum()==0:
        turn = target_w.abs().sum()
    else:
        pre_val = (1+ret_m_row.fillna(0.0)) * prev_w.fillna(0.0)
        pre_w = pre_val/(pre_val.sum() if pre_val.sum()>0 else 1.0)
        turn = (target_w.sub(pre_w).abs().sum())/2.0
    return float(turn), float(turn*one_way)

def wf_score(prices: pd.DataFrame, lookback_m:int, top_n:int, sma_win:int):
    monthly, fac = compute_factors(prices, lookback_m, sma_win)
    rets_m = monthly.pct_change().fillna(0.0)
    bench_m = rets_m["SPY"] if "SPY" in rets_m.columns else None
    idx = monthly.index
    if len(idx)<MIN_TRAIN_MONTHS+6: return -1e9, {}
    start_i = MIN_TRAIN_MONTHS
    prev_w=None; port=[]
    for i in range(start_i, len(idx)):
        t=idx[i]; tp=idx[i-1]
        scores = learn_asset_scores(fac, monthly, tp).dropna()
        picks = scores.nlargest(top_n).index
        hist = rets_m[picks].loc[:tp].iloc[-12:]
        w = minvar_weights(hist); w = scale_to_target_vol(w, hist)
        turn,cost = turnover_cost(prev_w, w.reindex(rets_m.columns).fillna(0.0), rets_m.loc[t])
        r = float((w*rets_m.loc[t,picks]).sum()) - cost
        port.append(r); prev_w = w.reindex(rets_m.columns).fillna(0.0)
    ret = pd.Series(port, index=idx[start_i:])
    m = perf_from_monthly(ret, bench_m.reindex_like(ret) if bench_m is not None else None)
    return score_obj(m), m

def search_best_params(prices: pd.DataFrame) -> Dict[str,int]:
    use_bo=False
    if USE_BAYES_OPT:
        try:
            from skopt import gp_minimize
            from skopt.space import Integer
            use_bo=True
        except Exception:
            use_bo=False
    if use_bo:
        space=[Integer(*PARAM_BOUNDS["LOOKBACK_M"], name="LOOKBACK_M"),
               Integer(*PARAM_BOUNDS["TOP_N"], name="TOP_N"),
               Integer(*PARAM_BOUNDS["SMA_WIN"], name="SMA_WIN")]
        def obj(x):
            pU = dict(zip(["LOOKBACK_M","TOP_N","SMA_WIN"], x))
            s,_ = wf_score(prices, int(pU["LOOKBACK_M"]), int(pU["TOP_N"]), int(pU["SMA_WIN"]))
            return -s
        res = gp_minimize(obj, space, n_calls=N_TRIALS, n_initial_points=N_INIT, random_state=42)
        return {k:int(v) for k,v in zip(["LOOKBACK_M","TOP_N","SMA_WIN"], res.x)}
    best=None; tried=0; since=0
    def sample():
        return dict(LOOKBACK_M=int(np.random.randint(*PARAM_BOUNDS["LOOKBACK_M"])),
                    TOP_N=int(np.random.randint(*PARAM_BOUNDS["TOP_N"])),
                    SMA_WIN=int(np.random.randint(*PARAM_BOUNDS["SMA_WIN"])))
    while tried<N_TRIALS:
        pU = sample()
        s,_ = wf_score(prices, int(pU["LOOKBACK_M"]), int(pU["TOP_N"]), int(pU["SMA_WIN"]))
        tried+=1
        if best is None or s>best[0]:
            best=(s,pU); since=0
        else:
            since+=1
            if since>=PATIENCE: break
    return best[1]

class AlpacaBroker:
    def __init__(self):
        from alpaca_trade_api import REST
        self.api = REST(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"),
                        base_url=os.getenv("ALPACA_BASE","https://paper-api.alpaca.markets"))
    def cash(self)->float:
        return float(self.api.get_account().cash)
    def positions(self)->Dict[str,float]:
        return {p.symbol: float(p.qty) for p in self.api.list_positions()}
    def submit_limit(self, symbol:str, qty:int, limit_price:float):
        if qty==0: return None
        side="buy" if qty>0 else "sell"
        o=self.api.submit_order(symbol=symbol, qty=abs(qty), side=side,
                                type="limit", time_in_force="day",
                                limit_price=round(float(limit_price), 2))
        return o
    def poll(self, sec=POLL_SEC): time.sleep(sec)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"last_rebalance": None, "params": None, "last_param_train_month": None}

def save_state(st): 
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(st,f,ensure_ascii=False,indent=2,default=_json_default)

def append_csv(path, row, header=None):
    new = not os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f); 
        if new and header: w.writerow(header)
        w.writerow(row)

def period_key(dt: pd.Timestamp, mode: str) -> str:
    if mode == "daily":  return dt.strftime("%Y-%m-%d")
    if mode == "weekly":
        y, w, _ = dt.isocalendar(); return f"{y}-W{int(w):02d}"
    return dt.strftime("%Y-%m")

def run_once():
    prices = download_prices(UNIVERSE, START_DATE, END_DATE)
    monthly = prices.resample("ME").last()
    rets_m  = monthly.pct_change().fillna(0.0)
    bench_m = rets_m["SPY"] if "SPY" in rets_m.columns else None

    state = load_state()
    latest_me = month_ends(prices.index)[-1]

    last_px_dt = prices.index[-1]
    cur_key = period_key(pd.Timestamp(last_px_dt), REBALANCE_MODE) if REBALANCE_MODE!="monthly" else latest_me.strftime("%Y-%m")

    # 选参（月节奏）
    need_search = False
    if not state.get("params"):
        need_search=True
    else:
        last_m = state.get("last_param_train_month")
        last_dt = pd.to_datetime((last_m or latest_me.strftime("%Y-%m"))+"-01") + pd.offsets.MonthEnd(0)
        if (((latest_me.year-last_dt.year)*12 + (latest_me.month-last_dt.month)) >= RETRAIN_EVERY_M):
            need_search=True
    if need_search:
        best = search_best_params(prices)
        best = {k:int(v) for k,v in best.items()}
        state["params"]=best
        state["last_param_train_month"]=latest_me.strftime("%Y-%m")
        with open(PARAMS_FILE,"w",encoding="utf-8") as f: json.dump(best,f,ensure_ascii=False,indent=2)
        print("[PARAM]", best, "month", state["last_param_train_month"])

    params = state["params"]
    lookback_m, top_n, sma_win = int(params["LOOKBACK_M"]), int(params["TOP_N"]), int(params["SMA_WIN"])

    # 当期是否已换仓
    if state.get("last_rebalance") == cur_key:
        px_today = prices.iloc[-1]
        broker = AlpacaBroker()
        pv = broker.cash() + sum(broker.positions().get(s,0.0)*px_today.get(s,np.nan) for s in UNIVERSE if s in px_today.index)
        print(f"[INFO] {px_today.name.date()} 资产估计: {pv:,.2f}（已换仓，mode={REBALANCE_MODE} key={cur_key} tag={TAG}）")
        return

    # 生成信号（上期末）
    monthly_p, fac = compute_factors(prices, lookback_m, sma_win)
    tp = monthly_p.index[-2]; t = monthly_p.index[-1]
    scores = learn_asset_scores(fac, monthly_p, tp).dropna()
    picks = scores.nlargest(top_n).index.tolist()

    hist = rets_m[picks].loc[:tp].iloc[-12:]
    w = minvar_weights(hist); w = scale_to_target_vol(w, hist)

    # —— 执行（市价化限价）
    px = last_trade_before(prices, t)
    px_sel = px.reindex(picks).dropna()
    broker = AlpacaBroker()
    cash = broker.cash(); pos = broker.positions()
    port_val = cash + sum(pos.get(sym,0.0)*px.get(sym,np.nan) for sym in UNIVERSE if sym in px.index)

    target_dollar = (w*port_val).reindex(px_sel.index).fillna(0.0)
    cur_dollar = pd.Series({sym: pos.get(sym,0.0)*px_sel[sym] for sym in px_sel.index}).fillna(0.0)
    diff = target_dollar - cur_dollar

    tot_now = port_val if port_val>0 else 1.0
    cur_w = (cur_dollar.reindex(px_sel.index).fillna(0.0)/tot_now).clip(0,1.0)
    tgt_w = (target_dollar.reindex(px_sel.index).fillna(0.0)/tot_now).clip(0,1.0)
    max_dev = float((cur_w - tgt_w).abs().max())
    if DEVIATION_THRESHOLD and max_dev <= DEVIATION_THRESHOLD:
        print(f"[SKIP] 偏离 {max_dev:.2%} ≤ 阈值 {DEVIATION_THRESHOLD:.2%}，跳过调仓（mode={REBALANCE_MODE}, key={cur_key}, tag={TAG}）")
        px_now = prices.loc[:t].iloc[-1]
        cash_now = broker.cash(); pos_now = broker.positions()
        pv_now = cash_now + sum(pos_now.get(sym,0.0)*px_now.get(sym,np.nan) for sym in UNIVERSE if sym in px_now.index)
        strat_eq = pv_now/INIT_CASH
        spy_eq = (px_now["SPY"]/prices["SPY"].iloc[0]) if "SPY" in px_now.index else ""
        append_csv(EQUITY_FILE, [t.strftime("%Y-%m-%d"), strat_eq, spy_eq], header=["date","strategy_eq","spy_eq"])
        return

    orders=[]
    for sym in px_sel.index:
        qty = int(round(diff[sym]/px_sel[sym]))
        if qty==0: continue
        limit = px_sel[sym]*(1+LIMIT_EPS if qty>0 else 1-LIMIT_EPS)
        try:
            broker.submit_limit(sym, qty, limit); orders.append((sym, qty, float(limit)))
        except Exception as e:
            print("[WARN] 下单失败:", sym, e)
    broker.poll()

    for sym, qty, price in orders:
        append_csv(TRADES_FILE, [t.strftime("%Y-%m-%d"), sym, qty, price], header=["date","ticker","qty","limit_price"])

    state["last_rebalance"] = cur_key; 
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(state,f,ensure_ascii=False,indent=2)

    px_now = prices.loc[:t].iloc[-1]
    cash_now = broker.cash(); pos_now = broker.positions()
    pv_now = cash_now + sum(pos_now.get(sym,0.0)*px_now.get(sym,np.nan) for sym in UNIVERSE if sym in px_now.index)
    strat_eq = pv_now/INIT_CASH
    spy_eq = (px_now["SPY"]/prices["SPY"].iloc[0]) if "SPY" in px_now.index else ""
    append_csv(EQUITY_FILE, [t.strftime("%Y-%m-%d"), strat_eq, spy_eq], header=["date","strategy_eq","spy_eq"])

    print(f"[REBAL] {t.date()} 选股:{picks}  资产:{pv_now:,.2f}  mode={REBALANCE_MODE} key={cur_key} tag={TAG}")

if __name__=="__main__":
    run_once()
