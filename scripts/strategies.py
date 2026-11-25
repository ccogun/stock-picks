"""전통 투자 전략(가치·배당·퀄리티·역추세·MACD·볼린저밴드)과 단기 스윙(1~5일) 스크리닝."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import MARKETS, fmt_price

SIG_LABEL = {"value": "가치(저PER·저PBR)", "dividend": "고배당", "quality": "퀄리티(고ROE·저부채)",
             "reversal": "역추세(RSI 과매도 탈출)", "macd": "MACD 골든크로스", "bollinger": "볼린저밴드 하단 반등",
             "swing": "단기 스윙"}


def _ind(df: pd.DataFrame) -> dict | None:
    from run import atr, ret, rsi
    c = df["Close"]
    if len(c) < 130:
        return None
    r = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    return {
        "c": float(c.iloc[-1]), "prev": float(c.iloc[-2]), "o": float(df["Open"].iloc[-1]),
        "h": float(df["High"].iloc[-1]), "l": float(df["Low"].iloc[-1]), "prev_h": float(df["High"].iloc[-2]),
        "ma20": float(ma20.iloc[-1]), "ma50": float(c.rolling(50).mean().iloc[-1]),
        "ma200": float(c.rolling(200, min_periods=120).mean().iloc[-1]),
        "rsi": float(r.iloc[-1]), "rsi_prev": float(r.iloc[-2]),
        "macd": float(macd.iloc[-1]), "macd_prev": float(macd.iloc[-2]),
        "sig": float(sig.iloc[-1]), "sig_prev": float(sig.iloc[-2]),
        "bb_lo": float((ma20 - 2 * sd20).iloc[-1]), "low3": float(df["Low"].tail(3).min()),
        "bb_lo3": float((ma20 - 2 * sd20).tail(3).min()), "low5": float(df["Low"].tail(5).min()),
        "atr": float(atr(df).iloc[-1]), "r63": ret(c, 63), "r20": ret(c, 20),
        "hi52": float(df["High"].tail(252).max()),
    }


def _liquid(df: pd.DataFrame, market: str) -> bool:
    v, c = df["Volume"].iloc[-21:-1], df["Close"].iloc[-21:-1]
    return float(v.mean() * c.mean()) >= MARKETS[market]["min_value"] and v.mean() > 0


def _pct_rank(vals: pd.Series, x: float) -> float:
    return float((vals < x).mean() * 100) if len(vals) and not math.isnan(x) else 50.0


def _stops(x: dict, kind: str) -> tuple[float, float]:
    c = x["c"]
    if kind == "fundamental":
        stop = min(max(c - 2.5 * x["atr"], c * 0.90), c * 0.95)
    elif kind == "swing":
        stop = min(max(x["l"], c - 1.0 * x["atr"], c * 0.94), c * 0.98)
    else:
        stop = min(max(x["low5"] * 0.99, c - 2 * x["atr"]), c * 0.97)
    return stop, c + 2 * (c - stop)


# ───────────────────────── 전통 전략 ─────────────────────────
def screen_strategies(market: str, universe: list[list[str]], data: dict, reg: dict, P: dict,
                      fund: dict) -> list[dict]:
    kr = MARKETS[market]["region"] == "KR"
    rs_all = pd.Series([x for t, _ in universe if (d := data.get(t)) is not None and len(d) > 63
                        and not math.isnan(x := float(d["Close"].iloc[-1] / d["Close"].iloc[-64] - 1))])
    out = []
    for ticker, name in universe:
        df = data.get(ticker)
        if df is None or not _liquid(df, market):
            continue
        x = _ind(df)
        if not x or x["rsi"] >= P["rsi_max"]:
            continue
        if kr and x["c"] / x["prev"] - 1 >= 0.25:
            continue   # 상한가 근처 제외
        f = fund.get(ticker) or {}
        up_today = x["c"] > x["prev"]
        rs_pct = _pct_rank(rs_all, x["r63"])
        sigs = []
        per, pbr, dy, roe, de = f.get("per"), f.get("pbr"), f.get("dy"), f.get("roe"), f.get("de")
        # 가치: 저PER·저PBR + 20일선 위로 올라선 날(떨어지는 칼날 회피)
        if per and pbr and 0 < per <= P["value_per_max_kr" if kr else "value_per_max_us"] \
                and 0 < pbr <= P["value_pbr_max_kr" if kr else "value_pbr_max_us"] \
                and x["c"] > x["ma20"] and up_today and 40 <= x["rsi"] <= 65:
            sigs.append(("value", f"PER {per:.1f}배, PBR {pbr:.2f}배로 저평가 구간",
                         60 - per + (2 - min(pbr, 2)) * 10))
        # 배당
        if dy and dy >= P["div_min_kr" if kr else "div_min_us"] and (f.get("payout") is None or f["payout"] < 0.8) \
                and x["c"] > x["ma20"] and x["c"] > x["ma200"] * 0.95:
            pay = f"배당성향 {f['payout'] * 100:.0f}%" if f.get("payout") else "배당성향 확인 필요"
            sigs.append(("dividend", f"배당수익률 {dy * 100:.1f}%, {pay}", dy * 1000))
        # 퀄리티
        if roe and roe >= P["quality_roe_min"] and (de is None or de < 1.0) and (f.get("om") is None or f["om"] > 0.1) \
                and x["c"] > x["ma50"] > x["ma200"] and x["rsi"] < 70:
            sigs.append(("quality", f"ROE {roe * 100:.0f}%, 부채비율 {de * 100:.0f}%" if de is not None
                         else f"ROE {roe * 100:.0f}%", roe * 100 + rs_pct / 5))
        # 역추세: RSI 30 아래에서 다시 위로 (장기 추세는 살아 있는 종목만)
        if x["rsi_prev"] < P["reversal_rsi"] <= x["rsi"] and x["c"] > x["ma200"] and up_today:
            sigs.append(("reversal", f"RSI {x['rsi_prev']:.0f} → {x['rsi']:.0f}, 과매도 구간 탈출", 50 + rs_pct / 4))
        # MACD 골든크로스 (50일선 위)
        if x["macd_prev"] <= x["sig_prev"] and x["macd"] > x["sig"] and x["c"] > x["ma50"]:
            sigs.append(("macd", "MACD선이 시그널선을 아래에서 위로 돌파(골든크로스)", 40 + rs_pct / 3))
        # 볼린저 하단 반등 (장기 상승 추세 안에서)
        if x["low3"] <= x["bb_lo3"] * 1.005 and x["c"] > x["bb_lo"] and up_today and x["c"] > x["ma200"]:
            sigs.append(("bollinger", "볼린저밴드 하단을 찍고 다시 밴드 안으로 반등", 45 + rs_pct / 4))
        if reg["regime"] == "하락":
            sigs = [s for s in sigs if s[0] in ("value", "dividend", "quality") or rs_pct >= 60]
        for s, why, sc in sigs:
            out.append({"ticker": ticker, "name": name, "signal": s, "why": why, "x": x, "f": f,
                        "rs_pct": rs_pct, "score": sc + P["weight"].get(s, 0)})
    # 같은 종목은 점수가 가장 높은 신호 하나만
    best: dict[str, dict] = {}
    for o in out:
        if o["ticker"] not in best or o["score"] > best[o["ticker"]]["score"]:
            best[o["ticker"]] = o
    # 신호별로 돌아가며 뽑기(학습 표본을 고르게)
    by_sig: dict[str, list] = {}
    for o in sorted(best.values(), key=lambda o: -o["score"]):
        by_sig.setdefault(o["signal"], []).append(o)
    order = sorted(by_sig, key=lambda s: -P["weight"].get(s, 0))
    ranked = []
    while any(by_sig.values()):
        for s in order:
            if by_sig[s]:
                ranked.append(by_sig[s].pop(0))
    return ranked


def strategy_reasons(o: dict, market: str, reg: dict) -> tuple[list[str], dict]:
    cur, label, x, f = MARKETS[market]["currency"], MARKETS[market]["label"], o["x"], o["f"]
    kind = "fundamental" if o["signal"] in ("value", "dividend", "quality") else "technical"
    stop, target = _stops(x, kind)
    c = x["c"]
    pos = "20일선·50일선 위" if c > x["ma20"] and c > x["ma50"] else ("20일선 위" if c > x["ma20"] else "20일선 아래")
    extra = []
    if f.get("per"):
        extra.append(f"PER {f['per']:.1f}")
    if f.get("roe"):
        extra.append(f"ROE {f['roe'] * 100:.0f}%")
    if f.get("dy"):
        extra.append(f"배당 {f['dy'] * 100:.1f}%")
    lines = [
        f"전략: {SIG_LABEL[o['signal']]} — {o['why']}",
        "재무: " + (", ".join(extra) if extra else "재무 지표 일부 미확인") + (f" · {f['sector']}" if f.get("sector") else ""),
        f"타이밍: 주가 {pos}, RSI {x['rsi']:.0f}, 3개월 수익률 {x['r63'] * 100:+.1f}% ({label} 감시 종목 중 상위 {100 - o['rs_pct']:.0f}%)",
        f"시장 국면: {label} {reg['regime']} — {reg['note']}",
        f"리스크 관리: 손절 {fmt_price(stop, cur)} ({(stop / c - 1) * 100:.1f}%), 1차 목표 {fmt_price(target, cur)} "
        f"({(target / c - 1) * 100:+.1f}%)",
    ]
    note = {"value": "저평가 종목이 20일선(주황) 위로 올라선 날을 매수 시점(▲)으로 잡았습니다.",
            "dividend": "배당 매력이 있는 종목이 20일선(주황) 위에서 버티는 구간입니다.",
            "quality": "실적이 우수한 종목이 50일선(파랑) 위 상승 추세에 있는 구간입니다.",
            "reversal": "RSI가 30 아래(과매도)로 떨어졌다가 다시 올라온 반등 초입입니다.",
            "macd": "단기 추세 지표(MACD)가 상승으로 돌아선 시점입니다.",
            "bollinger": "변동성 하단(볼린저밴드 하단)까지 밀렸다가 되돌아온 자리입니다."}[o["signal"]]
    return lines, {"stop": round(stop, 2), "target": round(target, 2),
                   "chartNote": note + " 빨간 점선(손절선) 아래로 종가가 내려가면 추천을 종료합니다."}


# ───────────────────────── 단기 스윙 (1~5일) ─────────────────────────
def screen_swing(market: str, universe: list[list[str]], data: dict, reg: dict, P: dict, frac: float) -> list[dict]:
    r20_all = pd.Series([x for t, _ in universe if (d := data.get(t)) is not None and len(d) > 21
                         and not math.isnan(x := float(d["Close"].iloc[-1] / d["Close"].iloc[-21] - 1))])
    out = []
    vol_min = P["swing_vol"] + (0.5 if reg["regime"] == "하락" else 0)
    for ticker, name in universe:
        df = data.get(ticker)
        if df is None or not _liquid(df, market):
            continue
        x = _ind(df)
        if not x:
            continue
        chg = x["c"] / x["prev"] - 1
        rng = x["h"] - x["l"]
        close_pos = (x["c"] - x["l"]) / rng if rng > 0 else 0
        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        vol_ratio = float(df["Volume"].iloc[-1]) / (avg_vol * max(frac, 0.05)) if avg_vol else 0
        r20_pct = _pct_rank(r20_all, x["r20"])
        if not (x["c"] > x["ma20"] > x["ma50"] and x["c"] > x["prev_h"] and 0.02 <= chg <= 0.12
                and close_pos >= 0.7 and vol_ratio >= vol_min and r20_pct >= 70 and x["rsi"] < 80):
            continue
        out.append({"ticker": ticker, "name": name, "signal": "swing", "x": x, "chg": chg, "close_pos": close_pos,
                    "vol_ratio": vol_ratio, "rs_pct": r20_pct,
                    "score": r20_pct + min(vol_ratio, 5) * 6 + close_pos * 10})
    return sorted(out, key=lambda o: -o["score"])


def swing_reasons(o: dict, market: str, reg: dict) -> tuple[list[str], dict]:
    cur, label, x = MARKETS[market]["currency"], MARKETS[market]["label"], o["x"]
    stop, target = _stops(x, "swing")
    c = x["c"]
    lines = [
        f"신호: 전일 고가 돌파, 오늘 {o['chg'] * 100:+.1f}% · 거래량 평소의 {o['vol_ratio']:.1f}배(장중 시간 보정)",
        f"힘: 당일 고가 부근에서 거래(일중 위치 상위 {(1 - o['close_pos']) * 100:.0f}%), 20일 수익률 상위 {100 - o['rs_pct']:.0f}%",
        "추세: 주가 > 20일선 > 50일선 단기 정배열",
        f"시장 국면: {label} {reg['regime']} — 단기 매매는 국면보다 손절 규율이 우선",
        f"계획: 1~5거래일 보유, 손절 {fmt_price(stop, cur)} ({(stop / c - 1) * 100:.1f}%), "
        f"목표 {fmt_price(target, cur)} ({(target / c - 1) * 100:+.1f}%) · 모의 검증 중",
    ]
    return lines, {"stop": round(stop, 2), "target": round(target, 2),
                   "chartNote": "거래량이 붙으며 전일 고가를 넘어선 자리(▲)입니다. 단기 매매라 손절선(빨간 점선)을 "
                                "이탈하거나 5거래일이 지나면 종료합니다."}
