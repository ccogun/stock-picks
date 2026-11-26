"""매시간 실행: 장이 열린 시장을 스캔해 차트 신호 종목을 추천하고, 모든 추천 종목의 성과를 추적한다.

사용법:
  python scripts/run.py                 # 지금 열린(또는 방금 마감한) 시장만 스캔 + 전체 추적
  python scripts/run.py --markets all   # 4개 시장 모두 스캔
  python scripts/run.py --track-only    # 스캔 없이 추적만
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd

from params import load_params
from common import (KST, NY, DATA, HOLD_DAYS, MARKETS, MAX_NEW_PER_DAY, GOOD_EXCESS, BAD_EXCESS, PRICES,
                    fmt_price, load_json, now_kst, region_recent, safe_name, save_json, session_phase,
                    session_state)

PICKS_FILE = DATA / "picks.json"
MARKET_FILE = DATA / "market.json"
LOG_FILE = DATA / "runs.json"


# ───────────────────────── 시세 수집 ─────────────────────────
def _kr_code(t: str) -> str | None:
    if t in ("^KS11", "^KQ11"):
        return t[1:]
    if t.endswith((".KS", ".KQ")):
        return t[:-3]
    return None


def download_kr(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """한국 종목·지수는 FinanceDataReader(네이버·KRX)로 받는다. 야후가 당일 봉을 빠뜨리는 문제 회피."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return {}
    days = {"2y": 760, "3mo": 100}.get(period, 400)
    start = (now_kst() - timedelta(days=days)).strftime("%Y-%m-%d")

    def one(t):
        try:
            df = fdr.DataReader(_kr_code(t), start)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            df = df[df["Close"] > 0]
            if len(df) >= 30:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return t, df.astype(float)
        except Exception:  # noqa: BLE001
            pass
        return t, None

    with ThreadPoolExecutor(8) as ex:
        return {t: df for t, df in ex.map(one, tickers) if df is not None}


def download(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    tickers = sorted(set(tickers))
    out: dict[str, pd.DataFrame] = download_kr([t for t in tickers if _kr_code(t)], period)
    tickers = [t for t in tickers if t not in out]   # 못 받은 것은 야후로
    for i in range(0, len(tickers), 150):
        chunk = tickers[i:i + 150]
        try:
            raw = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"[download] 실패 {chunk[:3]}…: {e}")
            continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df) >= 30:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    out[t] = df.astype(float)
            except Exception:  # noqa: BLE001
                pass
    return out


def download_ext(tickers: list[str]) -> dict[str, dict]:
    """미국 프리·애프터마켓 최신 가격 (5분봉, prepost=True). {ticker: {price, at, vol}}"""
    import yfinance as yf
    out: dict[str, dict] = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            raw = yf.download(chunk, period="2d", interval="5m", prepost=True, group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"[ext] 실패: {e}")
            continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Close"])
                if len(df):
                    ts = pd.Timestamp(df.index[-1])
                    out[t] = {"price": float(df["Close"].iloc[-1]), "at": ts.isoformat(),
                              "vol": float(df["Volume"].tail(36).sum())}
            except Exception:  # noqa: BLE001
                pass
    return out


# ───────────────────────── 지표 ─────────────────────────
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ret(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def regime(idx: pd.DataFrame) -> dict:
    c = idx["Close"]
    ma50, ma200 = c.rolling(50).mean().iloc[-1], c.rolling(200, min_periods=120).mean().iloc[-1]
    last = c.iloc[-1]
    if last > ma50 and ma50 > ma200:
        state, note = "상승", "지수가 50일·200일선 위 정배열 — 추세 추종 신호를 적극 반영"
    elif last < ma50 and ma50 < ma200:
        state, note = "하락", "지수가 50일선 아래 역배열 — 시장보다 강한 종목만 엄격히 선별"
    else:
        state, note = "조정", "지수가 50일선 부근에서 방향 탐색 — 돌파 신호 위주로 선별"
    peak = c.tail(252).max()
    return {"close": round(float(last), 2), "ma50": round(float(ma50), 2), "ma200": round(float(ma200), 2),
            "chg1d": round(ret(c, 1) * 100, 2), "chg20d": round(ret(c, 20) * 100, 2),
            "fromPeak": round((last / peak - 1) * 100, 2), "regime": state, "note": note}


# ───────────────────────── 차트 스크리닝 ─────────────────────────
def screen(market: str, universe: list[list[str]], data: dict[str, pd.DataFrame], idx: pd.DataFrame,
           reg: dict, frac: float) -> list[dict]:
    cfg = MARKETS[market]
    P = load_params()
    idx_r63 = ret(idx["Close"], 63)
    rows = []
    for ticker, name in universe:
        df = data.get(ticker)
        if df is None or len(df) < 130:
            continue
        c, v = df["Close"], df["Volume"]
        last = float(c.iloc[-1])
        ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
        ma200 = c.rolling(200, min_periods=120).mean()
        avg_vol = float(v.iloc[-21:-1].mean())
        value = avg_vol * float(c.iloc[-21:-1].mean())
        if not (value >= cfg["min_value"] and avg_vol > 0):
            continue
        if cfg["region"] == "KR" and last / float(c.iloc[-2]) - 1 >= 0.25:
            continue   # 상한가 근처는 실제로 사기 어려워 제외
        r = float(rsi(c).iloc[-1])
        a = float(atr(df).iloc[-1])
        hi52 = float(df["High"].tail(252).max())
        prev20_hi = float(df["High"].iloc[-21:-1].max())
        vol_ratio = float(v.iloc[-1]) / (avg_vol * frac)
        r63 = ret(c, 63)
        ma200_up = ma200.iloc[-1] > ma200.iloc[-21]
        trend = last > ma50.iloc[-1] > ma200.iloc[-1] and ma200_up
        near_high = last >= hi52 * 0.85
        if not (trend and near_high) or math.isnan(r63) or math.isnan(r):
            continue
        breakout = last > prev20_hi and vol_ratio >= P["breakout_vol"] and c.iloc[-1] > c.iloc[-2]
        low3 = float(df["Low"].tail(3).min())
        pullback = (low3 <= ma20.iloc[-1] * 1.01 and last > ma20.iloc[-1]
                    and P["pullback_rsi_lo"] <= r <= P["pullback_rsi_hi"]
                    and c.iloc[-1] > c.iloc[-2] and vol_ratio >= P["pullback_vol"])
        if not (breakout or pullback) or r >= P["rsi_max"]:
            continue
        rows.append(dict(ticker=ticker, name=name, last=last, rsi=r, atr=a, hi52=hi52, r63=r63,
                         rs=r63 - idx_r63, vol_ratio=vol_ratio, breakout=breakout,
                         ma20=float(ma20.iloc[-1]), ma50=float(ma50.iloc[-1]), ma200=float(ma200.iloc[-1])))
    if not rows:
        return []
    cand = pd.DataFrame(rows)
    # 상대강도 백분위(유니버스 안에서)
    all_rs = [ret(d["Close"], 63) for t, _ in universe if (d := data.get(t)) is not None and len(d) > 63]
    all_rs = pd.Series([x for x in all_rs if not math.isnan(x)])
    cand["rs_pct"] = cand["r63"].apply(lambda x: float((all_rs < x).mean() * 100) if len(all_rs) else 50.0)
    min_pct = P["rs_min_bear"] if reg["regime"] == "하락" else P["rs_min"]
    cand = cand[cand["rs_pct"] >= min_pct]
    if reg["regime"] == "하락" and not P["pullback_in_bear"]:
        cand = cand[cand["breakout"]]  # 약세장에서는 돌파 신호만
    w = P["weight"]
    cand["score"] = (cand["rs_pct"] + np.minimum(cand["vol_ratio"], 4) * 5
                     + np.where(cand["breakout"], w["breakout"], w["pullback"]))
    return cand.sort_values("score", ascending=False).to_dict("records")


def screen_ext(market: str, universe: list[list[str]], data: dict[str, pd.DataFrame], idx: pd.DataFrame,
               reg: dict, phase: str) -> list[dict]:
    """미국 시간외: 추세 종목 중 프리·애프터마켓 가격이 직전 고점을 갭으로 넘는 종목."""
    cfg = MARKETS[market]
    P = load_params()
    idx_r63 = ret(idx["Close"], 63)
    pre = []
    for ticker, name in universe:
        df = data.get(ticker)
        if df is None or len(df) < 130:
            continue
        c = df["Close"]
        last = float(c.iloc[-1])
        ma50 = float(c.rolling(50).mean().iloc[-1])
        ma200s = c.rolling(200, min_periods=120).mean()
        value = float(df["Volume"].iloc[-20:].mean() * c.iloc[-20:].mean())
        if value >= cfg["min_value"] and last > ma50 > ma200s.iloc[-1] and ma200s.iloc[-1] > ma200s.iloc[-21] \
                and last >= float(df["High"].tail(252).max()) * 0.85:
            pre.append((ticker, name))
    if not pre:
        return []
    ext = download_ext([t for t, _ in pre])
    all_rs = pd.Series([x for t, _ in universe if (d := data.get(t)) is not None and len(d) > 63
                        and not math.isnan(x := ret(d["Close"], 63))])
    rows = []
    for ticker, name in pre:
        e = ext.get(ticker)
        if not e:
            continue
        df = data[ticker]
        c = df["Close"]
        last_close = float(c.iloc[-1])
        hi20 = float(df["High"].tail(20).max())
        px = e["price"]
        gap = px / last_close - 1
        if gap < P["ext_gap_min"] or gap > P["ext_gap_max"] or px <= hi20:   # 과도한 급등은 추격하지 않음
            continue
        r63 = ret(c, 63)
        rs_pct = float((all_rs < r63).mean() * 100) if len(all_rs) else 50.0
        if rs_pct < (P["rs_min_bear"] if reg["regime"] == "하락" else P["rs_min"]):
            continue
        rows.append(dict(ticker=ticker, name=name, last=px, rsi=float(rsi(c).iloc[-1]), atr=float(atr(df).iloc[-1]),
                         hi52=float(df["High"].tail(252).max()), r63=r63, rs=r63 - idx_r63, rs_pct=rs_pct,
                         vol_ratio=0.0, breakout=True, gap=gap, ext=phase, extAt=e["at"],
                         ma20=float(c.rolling(20).mean().iloc[-1]), ma50=float(c.rolling(50).mean().iloc[-1])))
    rows.sort(key=lambda r: (r["rs_pct"] + min(r["gap"] * 100, 10) * 2), reverse=True)
    return rows


def chart_reasons(c: dict, market: str, reg: dict) -> tuple[list[str], dict]:
    cur = MARKETS[market]["currency"]
    label = MARKETS[market]["label"]
    last = c["last"]
    stop = max(last - 2 * c["atr"], c["ma50"] * 0.97) if c["breakout"] else max(last - 2 * c["atr"], c["ma50"] * 0.98)
    stop = min(stop, last * 0.97)
    target = last + 2 * (last - stop)
    if c.get("ext"):
        ph = "프리마켓" if c["ext"] == "pre" else "애프터마켓"
        sig = (f"{ph}에서 전일 종가 대비 {c['gap'] * 100:+.1f}% 갭으로 20일 고점 돌파 "
               f"(시간외 가격 기준, 정규장 거래량 확인 필요)")
    elif c["breakout"]:
        sig = f"20일 고점 돌파 + 거래량이 평소의 {c['vol_ratio']:.1f}배 (장중 경과 시간 보정)"
    else:
        sig = f"20일선까지 눌린 뒤 반등, 거래량 평소의 {c['vol_ratio']:.1f}배"
    lines = [
        f"신호: {sig}",
        f"추세: 주가 > 50일선 > 200일선 정배열, 200일선 상승 중 · 52주 고점 대비 {(last / c['hi52'] - 1) * 100:+.1f}%",
        f"상대강도: 3개월 {c['r63'] * 100:+.1f}% (지수 대비 {c['rs'] * 100:+.1f}%p) · {label} 감시 종목 중 상위 {100 - c['rs_pct']:.0f}%",
        f"시장 국면: {label} {reg['regime']} — {reg['note']}",
        f"리스크 관리: 손절 {fmt_price(stop, cur)} ({(stop / last - 1) * 100:.1f}%), 1차 목표 {fmt_price(target, cur)} "
        f"({(target / last - 1) * 100:+.1f}%), RSI {c['rsi']:.0f}",
    ]
    note = ("차트의 캔들은 일봉, 주황선은 20일선, 파란선은 50일선입니다. "
            + ("최근 20일 고점(점선)을 거래량을 동반해 넘어선 자리가 추천 시점(▲)이며, " if c["breakout"]
               else "주가가 20일선까지 내려왔다가 다시 올라선 자리가 추천 시점(▲)이며, ")
            + "빨간 점선(손절선) 아래로 종가가 내려가면 추천을 종료합니다.")
    return lines, {"stop": round(stop, 2), "target": round(target, 2), "chartNote": note}


# ───────────────────────── 성과 추적 ─────────────────────────
def close_on(df: pd.DataFrame, date_str: str) -> float | None:
    d = pd.Timestamp(date_str)
    s = df["Close"][df.index <= d]
    return float(s.iloc[-1]) if len(s) else None


MIN_DAYS_FOR_VERDICT = 3   # 3거래일 전에는 손절 이탈이 아니면 '판단 중'
SNAPSHOTS = (5, 10, 20)     # 학습용: 5·10·20거래일 시점의 지수 대비 초과수익 기록


def entry_from(market: str, at) -> str | None:
    """정규장 밖에서 한 추천은 '그 뒤 첫 정규장 시가'를 진입가로 쓴다. 기준 날짜를 돌려준다(정규장 중이면 None)."""
    from datetime import datetime as _dt, timedelta as _td
    if isinstance(at, str):
        at = _dt.fromisoformat(at)
    ph = session_phase(MARKETS[market]["region"], at)
    if ph == "regular":
        return None
    from datetime import time as _t
    region = MARKETS[market]["region"]
    local = at.astimezone(KST if region == "KR" else NY)
    start = _t(9, 0) if region == "KR" else _t(9, 30)
    d = local.replace(tzinfo=None)
    if local.time() >= start:   # 장 마감 후면 다음 날 이후 첫 거래일, 장 시작 전이면 그날 정규장
        d += _td(days=1)
    return d.strftime("%Y-%m-%d")


def settle_entry(p: dict, df: pd.DataFrame, idx: pd.DataFrame | None) -> bool:
    """진입 대기 중이면 첫 정규장 시가로 진입가를 확정한다. 아직 그 거래일이 없으면 False."""
    if p.get("entry") != "pending":
        return True
    bars = df[df.index >= pd.Timestamp(p["entryFrom"])]
    if not len(bars):
        return False
    d, b = bars.index[0], bars.iloc[0]
    old, new = p.get("recPrice"), round(float(b["Open"]), 4)
    p["recPrice"], p["recDate"] = new, d.strftime("%Y-%m-%d")
    if idx is not None:
        io = idx[idx.index >= d]
        p["recIndex"] = float(io["Open"].iloc[0]) if len(io) else close_on(idx, p["recDate"])
    if old and float(old) > 0:      # 손절·목표가는 진입가에 맞춰 같은 비율로 이동
        r = new / float(old)
        for k in ("stop", "target"):
            if p.get(k):
                p[k] = round(float(p[k]) * r, 4)
    note = f"추천 시각이 정규장 밖이라 {p['recDate']} 정규장 시가로 진입가 확정"
    prev = df[df.index < d]["Close"]
    if MARKETS[p["market"]]["region"] == "KR" and len(prev) and new >= float(prev.iloc[-1]) * 1.29:
        note += " (시초가 상한가 — 실제로는 사기 어려웠을 수 있음)"
    p["entry"], p["entryNote"] = "open", note
    p.pop("snap", None)
    return True


def track(p: dict, df: pd.DataFrame | None, idx: pd.DataFrame | None) -> None:
    if df is None or len(df) == 0:
        return
    if not settle_entry(p, df, idx):
        p["daysHeld"], p["verdict"] = 0, "진입 대기"
        for k in ("ret", "excess", "benchRet", "maxRet", "minRet"):
            p.pop(k, None)
        p["lastPrice"], p["lastDate"] = round(float(df["Close"].iloc[-1]), 4), df.index[-1].strftime("%Y-%m-%d")
        return
    P = load_params()
    if not p.get("recPrice"):
        p["recPrice"] = round(close_on(df, p["recDate"]) or float(df["Close"].iloc[-1]), 4)
    if not p.get("stop"):   # 손절·목표가를 안 준 추천은 기본값(학습으로 조정)
        p["stop"] = round(float(p["recPrice"]) * (1 - P["investor_stop"]), 4)
    if not p.get("target"):
        p["target"] = round(float(p["recPrice"]) * (1 + P["investor_target"]), 4)
    if idx is not None and not p.get("recIndex"):
        p["recIndex"] = close_on(idx, p["recDate"])
    last_date = df.index[-1]
    if last_date < pd.Timestamp(p["recDate"]) and p.get("lastDate"):
        return   # 시세가 추천일보다 오래됨(데이터 지연) — 이전 값 유지
    after = df[df.index >= pd.Timestamp(p["recDate"])]
    last = float(df["Close"].iloc[-1])
    rp = float(p["recPrice"])
    p["lastPrice"] = round(last, 4)
    p["lastDate"] = last_date.strftime("%Y-%m-%d")
    p["ret"] = round((last / rp - 1) * 100, 2)
    if len(after):
        p["maxRet"] = round((float(after["High"].max()) / rp - 1) * 100, 2)
        p["minRet"] = round((float(after["Low"].min()) / rp - 1) * 100, 2)
    if idx is not None and p.get("recIndex"):
        bench = close_on(idx, p["lastDate"])          # 종목과 같은 날짜의 지수로 비교
        if bench:
            p["benchRet"] = round((bench / p["recIndex"] - 1) * 100, 2)
            p["excess"] = round(p["ret"] - p["benchRet"], 2)
    days = max(len(after) - 1, 0)
    p["daysHeld"] = days
    snaps = p.setdefault("snap", {})
    for n in SNAPSHOTS:
        if days >= n and str(n) not in snaps and len(after) > n and idx is not None and p.get("recIndex"):
            d_n = after.index[n].strftime("%Y-%m-%d")
            px, bx = float(after["Close"].iloc[n]), close_on(idx, d_n)
            if bx:
                snaps[str(n)] = round((px / rp - 1) * 100 - (bx / p["recIndex"] - 1) * 100, 2)
    if p.get("status") != "active":
        return
    exc = p.get("excess", p["ret"]) / 100
    hold = 5 if p["method"] == "swing" else HOLD_DAYS
    stop_hit = p.get("stop") and last < float(p["stop"])
    target_hit = p.get("target") and len(after) and float(after["High"].max()) >= float(p["target"])
    if stop_hit:
        p["status"], p["verdict"], p["closeReason"] = "closed", "아쉬운 추천", "손절선 이탈"
    elif days >= hold:
        p["status"] = "closed"
        p["verdict"] = "잘한 추천" if exc > 0 else "아쉬운 추천"
        p["closeReason"] = f"평가 기간 {hold}거래일 종료"
    elif target_hit:
        p["verdict"] = "잘한 추천 (목표 도달)"
    elif days < (1 if p["method"] == "swing" else MIN_DAYS_FOR_VERDICT):
        p["verdict"] = "판단 중"
    elif exc >= GOOD_EXCESS:
        p["verdict"] = "잘한 추천"
    elif exc <= BAD_EXCESS:
        p["verdict"] = "아쉬운 추천"
    else:
        p["verdict"] = "판단 중"
    if p["status"] == "closed":
        p["closedAt"] = p["lastDate"]
        p["outcome"] = p.get("excess")   # 최종 지수 대비 초과수익(학습용)


def freeze_closed(p: dict, df: pd.DataFrame, idx: pd.DataFrame | None) -> None:
    """종료일 종가 기준으로 최종 성과를 고정한다."""
    end = pd.Timestamp(p.get("closedAt") or p.get("lastDate"))
    upto = df[df.index <= end]
    if not len(upto) or not p.get("recPrice"):
        return
    rp = float(p["recPrice"])
    last = float(upto["Close"].iloc[-1])
    p["lastPrice"], p["lastDate"] = round(last, 4), upto.index[-1].strftime("%Y-%m-%d")
    p["exitPrice"] = p["lastPrice"]
    p["ret"] = round((last / rp - 1) * 100, 2)
    held = upto[upto.index >= pd.Timestamp(p["recDate"])]
    if len(held):
        p["maxRet"] = round((float(held["High"].max()) / rp - 1) * 100, 2)
        p["minRet"] = round((float(held["Low"].min()) / rp - 1) * 100, 2)
        p["daysHeld"] = max(len(held) - 1, 0)
    if idx is not None and p.get("recIndex"):
        b = close_on(idx, p["lastDate"])
        if b:
            p["benchRet"] = round((b / p["recIndex"] - 1) * 100, 2)
            p["excess"] = round(p["ret"] - p["benchRet"], 2)
            p["outcome"] = p["excess"]
    p["frozen"] = True


def describe_chart(df: pd.DataFrame, currency: str) -> str:
    """투자자 방식 추천처럼 차트 설명이 없는 종목에 현재 차트 상태를 글로 붙인다."""
    c = df["Close"]
    last = float(c.iloc[-1])
    ma20, ma50 = float(c.rolling(20).mean().iloc[-1]), float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200, min_periods=120).mean().iloc[-1]) if len(c) >= 120 else float("nan")
    r = float(rsi(c).iloc[-1])
    hi = float(df["High"].tail(252).max())
    if last > ma20 > ma50:
        pos = "20일선·50일선 위에서 오르는 상승 추세"
    elif last > ma50:
        pos = "50일선 위에서 숨 고르는 중"
    elif last > ma20:
        pos = "20일선을 회복했지만 50일선 아래(반등 시도)"
    else:
        pos = "20일선·50일선 아래(하락 추세, 분할 접근 필요)"
    long = "" if math.isnan(ma200) else (" · 200일선 위" if last > ma200 else " · 200일선 아래")
    heat = "과열권" if r >= 70 else ("침체권" if r <= 30 else "중립")
    return (f"차트 상태: {pos}{long}. RSI {r:.0f}({heat}), 52주 고점 대비 {(last / hi - 1) * 100:+.1f}%. "
            f"주황선 20일선({fmt_price(ma20, currency)}), 파란선 50일선({fmt_price(ma50, currency)}), ▲가 추천 시점입니다.")


def write_prices(ticker: str, df: pd.DataFrame, keep: int = 160) -> None:
    tail = df.tail(keep)
    rows = [[d.strftime("%Y-%m-%d"), *(round(float(x), 4) for x in r)]
            for d, r in zip(tail.index, tail[["Open", "High", "Low", "Close", "Volume"]].values)]
    save_json(PRICES / f"{safe_name(ticker)}.json", {"t": ticker, "d": rows})


def market_date(market: str, at) -> str:
    """추천일은 그 시장 현지 날짜로 기록한다(미국 종목을 한국 새벽에 추천해도 미국 거래일 기준)."""
    from datetime import datetime as _dt
    if isinstance(at, str):
        at = _dt.fromisoformat(at)
    tz = KST if MARKETS[market]["region"] == "KR" else NY
    return at.astimezone(tz).strftime("%Y-%m-%d")


def ingest_inbox(picks: list[dict]) -> list[str]:
    """Claude가 docs/data/inbox/ 에 넣은 투자자 방식 추천을 picks에 합친다."""
    added = []
    inbox = DATA / "inbox"
    for f in sorted(inbox.glob("*.json")) if inbox.exists() else []:
        items = load_json(f, [])
        for it in items if isinstance(items, list) else [items]:
            try:
                m = it["market"]
                rec_at = it.get("recAt") or now_kst().isoformat(timespec="minutes")
                day = market_date(m, rec_at) if m in MARKETS else rec_at[:10]
                method = it.get("method", "investor") if it.get("method") in ("investor", "combined") else "investor"
                active = {p["ticker"] for p in picks if p["status"] == "active"
                          and (method != "combined" or p["method"] == "combined")}
                n_day = sum(1 for p in picks if p["market"] == m and p["method"] == method and p["recDate"] == day)
                if m not in MARKETS or it["ticker"] in active or n_day >= MAX_NEW_PER_DAY:
                    print(f"[inbox] 건너뜀: {it.get('name')} ({m})")
                    continue
                if method == "combined":   # 종합 추천은 오늘 후보 목록에 있는 종목만
                    cm = (load_json(DATA / "candidates.json", {}).get("markets", {}).get(m) or {}).get("items", [])
                    cand = next((c for c in cm if c["ticker"] == it["ticker"]), None)
                    if cand is None:
                        print(f"[inbox] 건너뜀(종합 후보 아님): {it.get('name')}")
                        continue
                p = {"id": f"{day}-{m}-{method}-{it['ticker']}", "market": m, "method": method,
                     "ticker": it["ticker"], "name": it["name"], "currency": MARKETS[m]["currency"],
                     "recAt": rec_at, "recDate": day, "recPrice": it.get("recPrice"),
                     "reasons": [str(r) for r in it["reasons"]][:5], "source": it.get("source", ""),
                     "status": "active", "verdict": "판단 중",
                     "features": {"style": it.get("style", "기타"),
                                  "regime": (load_json(MARKET_FILE, {}).get(m) or {}).get("regime")}}
                for k in ("stop", "target"):
                    if it.get(k):
                        p[k] = float(it[k])
                if method == "combined":
                    p.setdefault("stop", cand["stop"]); p.setdefault("target", cand["target"])
                    p["recPrice"] = p["recPrice"] or cand["price"]
                    p["features"].update(signal="+".join(cand["signals"]), signals=cand["signals"],
                                         nSignals=len(cand["signals"]), rsPct=cand["rs_pct"])
                    if cand.get("chartNote"):
                        p["chartNote"] = cand["chartNote"]
                if (ef := entry_from(m, rec_at)):
                    p.update(entry="pending", entryFrom=ef, verdict="진입 대기")
                if MARKETS[m]["region"] == "KR":
                    from flows import FLOW_FILE, flow_tag
                    p["features"]["flow"] = flow_tag(load_json(FLOW_FILE, {}).get(it["ticker"]))
                    if p["features"]["flow"] == "동반 순매도":
                        print(f"[inbox] 건너뜀(외국인·기관 동반 순매도): {it.get('name')}")
                        continue
                picks.append(p)
                added.append(f"{m}:{it['name']}")
            except (KeyError, TypeError) as e:
                print(f"[inbox] 형식 오류 {f.name}: {e}")
        f.unlink()
    return added


# ───────────────────────── 메인 ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="auto")
    ap.add_argument("--track-only", action="store_true")
    args = ap.parse_args()

    now = now_kst()
    today = now.strftime("%Y-%m-%d")
    db = load_json(PICKS_FILE, {"picks": []})
    picks: list[dict] = db["picks"]
    mkt_info = load_json(MARKET_FILE, {})
    inbox_added = ingest_inbox(picks)

    if args.markets == "all":
        scan = list(MARKETS)
    elif args.markets == "auto":
        scan = [m for m, c in MARKETS.items() if region_recent(c["region"], now)]
    else:
        scan = [m for m in args.markets.split(",") if m in MARKETS]
    if args.track_only:
        scan = []

    # 지수는 항상 받는다
    idx_data = download([c["index"] for c in MARKETS.values()], period="2y")
    for t in [c["index"] for c in MARKETS.values() if c["index"] not in idx_data]:
        idx_data.update(download([t], period="2y"))   # 빠진 지수는 한 번 더
    for m, c in MARKETS.items():
        if c["index"] in idx_data:
            mkt_info[m] = {**mkt_info.get(m, {}), **regime(idx_data[c["index"]]), "label": c["label"],
                           "phase": session_phase(c["region"], now),
                           "updated": now.isoformat(timespec="minutes")}
            write_prices(c["index"], idx_data[c["index"]])
        else:
            print(f"[index] {m} 지수 시세 없음 — 이전 값 유지")
            mkt_info.setdefault(m, {})["phase"] = session_phase(c["region"], now)

    log = {"at": now.isoformat(timespec="minutes"), "scanned": scan, "added": list(inbox_added)}
    all_cands: dict = load_json(DATA / "candidates.json", {}).get("markets", {})
    if scan:
        from universe import get_universe
        uni = get_universe()
        for m in scan:
            cfg = MARKETS[m]
            idx = idx_data.get(cfg["index"])
            if idx is None:
                continue
            today = market_date(m, now)   # 이 시장 현지 날짜
            open_, frac = session_state(cfg["region"], now)
            phase = session_phase(cfg["region"], now)
            data = download([t for t, _ in uni[m]], period="1y")
            print(f"[{m}] 시세 {len(data)}/{len(uni[m])}종목, 구간={phase}, 경과={frac:.2f}")
            if cfg["region"] == "US" and phase in ("pre", "after"):
                cands = screen_ext(m, uni[m], data, idx, mkt_info[m], phase)
            else:
                cands = screen(m, uni[m], data, idx, mkt_info[m], frac if open_ else 1.0)
            flows = {}
            if cfg["region"] == "KR":
                from flows import flow_tag, refresh as refresh_flows
                flows = refresh_flows([t for t, _ in uni[m]])
                fb = load_params()["flow_bonus"]
                for c in cands:
                    c["flow"] = flow_tag(flows.get(c["ticker"]))
                    c["score"] = c.get("score", 0) + (fb if c["flow"] == "동반 순매수" else 0)
                # 외국인·기관이 5일 동안 둘 다 순매도한 종목은 제외
                cands = [c for c in cands if c["flow"] != "동반 순매도"]
                cands.sort(key=lambda c: -c.get("score", 0))
            active = {p["ticker"] for p in picks if p["status"] == "active"}  # 시장 중복 추천 방지
            added_today = sum(1 for p in picks if p["market"] == m and p["method"] == "chart"
                              and p["recDate"] == today)
            for c in cands:
                if added_today >= MAX_NEW_PER_DAY:
                    break
                if c["ticker"] in active:
                    continue
                reasons, extra = chart_reasons(c, m, mkt_info[m])
                p = {"id": f"{today}-{m}-chart-{c['ticker']}", "market": m, "method": "chart",
                     "ticker": c["ticker"], "name": c["name"], "currency": cfg["currency"],
                     "recAt": now.isoformat(timespec="minutes"), "recDate": today,
                     "recPrice": round(c["last"], 4), "recIndex": float(idx["Close"].iloc[-1]),
                     "reasons": reasons, "source": "차트 분석(추세·돌파·상대강도)",
                     "features": {"signal": ("ext" if c.get("ext") else ("breakout" if c["breakout"] else "pullback")),
                                  "regime": mkt_info[m]["regime"], "rsPct": round(c["rs_pct"], 1),
                                  "volRatio": round(c["vol_ratio"], 2), "rsi": round(c["rsi"], 1),
                                  "gap": round(c.get("gap", 0) * 100, 2)},
                     "status": "active", "verdict": "판단 중", **extra}
                if (ef := entry_from(m, now)):
                    p.update(entry="pending", entryFrom=ef, verdict="진입 대기")
                if c.get("flow"):
                    p["features"]["flow"] = c["flow"]
                    p["reasons"][3] += f" · 수급(5일): 외국인·기관 {c['flow']}"
                picks.append(p)
                write_prices(c["ticker"], data[c["ticker"]])
                active.add(c["ticker"])
                added_today += 1
                log["added"].append(f"{m}:{c['name']}")
            mkt_info[m]["lastScan"] = now.isoformat(timespec="minutes")
            mkt_info[m]["candidates"] = len(cands)

            # 전통 전략 + 단기 스윙 (시간외 미국은 정규장 가격이 아니라 건너뜀)
            chart_all = list(cands)
            ext_us = cfg["region"] == "US" and phase in ("pre", "after")
            from fundamentals import refresh
            from strategies import screen_strategies, screen_swing, strategy_reasons, swing_reasons
            P = load_params()
            fund = refresh([t for t, _ in uni[m]])
            groups = [] if ext_us else [
                ("strategy", screen_strategies(m, uni[m], data, mkt_info[m], P, fund), strategy_reasons, "전통 투자 전략"),
                ("swing", screen_swing(m, uni[m], data, mkt_info[m], P, frac if open_ else 1.0)
                 if phase == "regular" else [], swing_reasons, "단기 스윙(1~5일, 모의 검증)")]
            if cfg["region"] == "KR":   # 전통 전략·단기 스윙도 동반 순매도 종목 제외
                groups = [(mth, [o for o in lst if flow_tag(flows.get(o["ticker"])) != "동반 순매도"], fn, sr)
                          for mth, lst, fn, sr in groups]
            for method, lst, reason_fn, src in groups:
                n_today = sum(1 for p in picks if p["market"] == m and p["method"] == method and p["recDate"] == today)
                for o in lst:
                    if n_today >= MAX_NEW_PER_DAY:
                        break
                    if o["ticker"] in active:
                        continue
                    reasons, extra = reason_fn(o, m, mkt_info[m])
                    x = o["x"]
                    picks.append({"id": f"{today}-{m}-{method}-{o['ticker']}", "market": m, "method": method,
                                  "ticker": o["ticker"], "name": o["name"], "currency": cfg["currency"],
                                  "recAt": now.isoformat(timespec="minutes"), "recDate": today,
                                  "recPrice": round(x["c"], 4), "recIndex": float(idx["Close"].iloc[-1]),
                                  "reasons": reasons, "source": src,
                                  "features": {"signal": o["signal"], "regime": mkt_info[m]["regime"],
                                               "rsPct": round(o["rs_pct"], 1), "rsi": round(x["rsi"], 1)},
                                  "status": "active", "verdict": "판단 중", **extra})
                    if (ef := entry_from(m, now)):
                        picks[-1].update(entry="pending", entryFrom=ef, verdict="진입 대기")
                    if cfg["region"] == "KR":
                        tag = flow_tag(flows.get(o["ticker"]))
                        picks[-1]["features"]["flow"] = tag
                        picks[-1]["reasons"][3] += f" · 수급(5일): 외국인·기관 {tag}"
                    write_prices(o["ticker"], data[o["ticker"]])
                    active.add(o["ticker"])
                    n_today += 1
                    log["added"].append(f"{m}:{o['name']}({method})")

            # ── 종합 추천: 투자자 근거 + 차트(기술적) 근거가 모두 있는 종목 ──
            from combined import build_candidates, auto_combined
            tech = {"chart": chart_all, "strategy": groups[0][1] if groups else [],
                    "swing": groups[1][1] if len(groups) > 1 else []}
            cands_c = build_candidates(m, tech, fund, flows if cfg["region"] == "KR" else {}, mkt_info[m], data,
                                       {"chart": chart_reasons, "strategy": strategy_reasons, "swing": swing_reasons})
            all_cands[m] = {"phase": phase, "items": cands_c}
            for p in auto_combined(m, cands_c, picks, today, now, idx, mkt_info[m]):
                if (ef := entry_from(m, now)):
                    p.update(entry="pending", entryFrom=ef, verdict="진입 대기")
                picks.append(p)
                write_prices(p["ticker"], data[p["ticker"]])
                log["added"].append(f"{m}:{p['name']}(종합)")

    save_json(DATA / "candidates.json", {"updated": now.isoformat(timespec="minutes"), "markets": all_cands})

    # 추적: 진행 중 + 최근 종료 종목
    need = [p["ticker"] for p in picks if p["status"] == "active" or not p.get("lastPrice")]
    pdata = download(need, period="1y") if need else {}
    for p in picks:
        idx = idx_data.get(MARKETS[p["market"]]["index"])
        if p["status"] == "closed":
            # 종료된 추천은 종료일 기준으로 고정 (같은 종목이 다시 추천돼도 이전 기록이 바뀌지 않게)
            if not p.get("frozen"):
                df_c = pdata.get(p["ticker"])
                if df_c is None:
                    saved = load_json(PRICES / f"{safe_name(p['ticker'])}.json", {}).get("d")
                    if saved:
                        df_c = pd.DataFrame([r[1:] for r in saved], index=pd.to_datetime([r[0] for r in saved]),
                                            columns=["Open", "High", "Low", "Close", "Volume"])
                if df_c is not None:
                    freeze_closed(p, df_c, idx)
            continue
        if p["ticker"] in pdata:
            track(p, pdata[p["ticker"]], idx)
            if p["method"] != "chart" and p["status"] == "active":
                p["chartNote"] = describe_chart(pdata[p["ticker"]], p["currency"])
            write_prices(p["ticker"], pdata[p["ticker"]])

    # 미국 시간외 가격(참고용): 프리·애프터마켓이면 진행 중 미국 종목에 시간외 가격 표시
    us_phase = session_phase("US", now)
    us_active = [p for p in picks if p["status"] == "active" and MARKETS[p["market"]]["region"] == "US"]
    if us_phase in ("pre", "after") and us_active:
        ext = download_ext([p["ticker"] for p in us_active])
        for p in us_active:
            e = ext.get(p["ticker"])
            if e and p.get("recPrice"):
                p["ext"] = {"phase": us_phase, "price": round(e["price"], 4), "at": e["at"],
                            "ret": round((e["price"] / float(p["recPrice"]) - 1) * 100, 2)}
    elif us_phase == "regular":
        for p in us_active:
            p.pop("ext", None)

    db = {"updated": now.isoformat(timespec="minutes"), "picks": picks}
    save_json(PICKS_FILE, db)
    save_json(MARKET_FILE, mkt_info)
    runs = load_json(LOG_FILE, [])
    runs = (runs + [log])[-200:]
    save_json(LOG_FILE, runs)
    print(f"완료: 스캔 {scan}, 신규 {log['added']}, 추적 {len(pdata)}종목")


if __name__ == "__main__":
    main()
