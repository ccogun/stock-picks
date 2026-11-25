"""지지 가격대 찾는 방법 비교 (미래 정보를 쓰지 않는 워크포워드 검증).

각 날짜 t에는 t 이전에 '확정된' 정보로만 지지선을 만든다(스윙 저점은 오른쪽 N일이 지나야 확정).
지지선에 위에서 처음 닿은 날 지정가(지지선 +1%)로 샀다고 가정하고 10·20거래일 뒤 성과를 본다.

비교 방법
- piv5 / piv10 / piv20 : 좌우 N일 최저 스윙 저점을 ±3%로 묶어 여러 번 모인 가격대
- hvn                  : 최근 1년 거래량 분포(볼륨 프로파일)에서 거래가 몰린 가격대
- ma200 / ma50         : 이동평균선
- piv10_hvn            : 스윙 저점 가격대 + 거래량 밀집이 겹치는 곳(합류)
- dip (대조군)         : 지지선 없이 최근 5일 저점보다 3% 이상 밀린 날 종가 매수 — '그냥 눌림 매수'와 비교용
결과: docs/data/sr_compare.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from patterns import KR, KST, US, DATA  # noqa: E402

H1, H2 = 10, 20
STEP = 5          # 지지선 재계산 간격(거래일)
GAP = 10          # 같은 방법·종목 이벤트 사이 최소 간격


def swing_lows(low: np.ndarray, n: int, upto: int) -> list[float]:
    """upto 시점까지 확정된 스윙 저점 (i+n <= upto-1)."""
    out = []
    for i in range(n, upto - n):
        w = low[i - n:i + n + 1]
        if low[i] == w.min():
            out.append(low[i])
    return out


def cluster(vals: list[float], band: float = 0.03, min_n: int = 3) -> list[float]:
    vals = sorted(vals)
    groups: list[list[float]] = []
    for v in vals:
        if groups and v <= np.mean(groups[-1]) * (1 + 2 * band):
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups if len(g) >= min_n]


def hvn_levels(df: pd.DataFrame, upto: int, look: int = 250) -> list[float]:
    d = df.iloc[max(0, upto - look):upto]
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if hi <= lo:
        return []
    edges = np.exp(np.arange(np.log(lo), np.log(hi) + 0.01, 0.01))   # 1% 간격
    vol = np.zeros(len(edges))
    for l, h, v in zip(d["Low"].to_numpy(), d["High"].to_numpy(), d["Volume"].to_numpy()):
        a, b = np.searchsorted(edges, l), max(np.searchsorted(edges, h), np.searchsorted(edges, l) + 1)
        vol[a:b] += v / (b - a)
    if vol.max() <= 0:
        return []
    sm = np.convolve(vol, np.ones(3) / 3, mode="same")
    thr = np.quantile(sm, 0.8)
    return [float(edges[i]) for i in range(1, len(sm) - 1)
            if sm[i] >= thr and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]]


def evaluate_ticker(df: pd.DataFrame) -> dict[str, list[dict]]:
    low, close = df["Low"].to_numpy(), df["Close"].to_numpy()
    n = len(df)
    ma = {k: pd.Series(close).rolling(k).mean().to_numpy() for k in (50, 200)}
    res: dict[str, list[dict]] = {k: [] for k in
                                  ("piv5", "piv10", "piv20", "hvn", "ma50", "ma200", "piv10_hvn", "dip")}
    last_ev = {k: -999 for k in res}
    levels: dict[str, list[float]] = {}
    start = 260
    for t in range(start, n - H2):
        if (t - start) % STEP == 0:
            levels["piv5"] = cluster(swing_lows(low, 5, t), min_n=3)
            levels["piv10"] = cluster(swing_lows(low, 10, t), min_n=3)
            levels["piv20"] = cluster(swing_lows(low, 20, t), min_n=2)
            levels["hvn"] = hvn_levels(df, t)
            levels["piv10_hvn"] = [z for z in levels["piv10"] if any(abs(h / z - 1) <= 0.02 for h in levels["hvn"])]
        prev_min = low[t - 5:t].min()

        def record(k, entry):
            if t - last_ev[k] < GAP:
                return
            last_ev[k] = t
            fut_lo = low[t + 1:t + H2 + 1]
            fut_cl = close[t + 1:t + H2 + 1]
            up = np.where(df["High"].to_numpy()[t + 1:t + H2 + 1] >= entry * 1.05)[0]
            dn = np.where(fut_lo <= entry * 0.95)[0]
            bounce = len(up) > 0 and (len(dn) == 0 or up[0] < dn[0])
            res[k].append({"r10": close[t + H1] / entry - 1, "r20": close[t + H2] / entry - 1,
                           "mdd": fut_lo.min() / entry - 1, "bounce": bounce,
                           "broke": bool((fut_cl < entry * 0.94).any()), "year": df.index[t].year})

        for k in ("piv5", "piv10", "piv20", "hvn", "piv10_hvn"):
            for L in levels.get(k, []):
                if prev_min > L * 1.03 and low[t] <= L * 1.01:
                    record(k, L * 1.01)
                    break
        for k, m in (("ma50", ma[50]), ("ma200", ma[200])):
            L = m[t]
            if not np.isnan(L) and np.all(low[t - 5:t] > m[t - 5:t] * 1.03) and low[t] <= L * 1.01:
                record(k, L * 1.01)
        if low[t] <= prev_min * 0.97:
            record("dip", close[t])
    return res


def summarize(evs: list[dict], base: float) -> dict:
    if not evs:
        return {"n": 0}
    r20 = np.array([e["r20"] for e in evs]) * 100
    r10 = np.array([e["r10"] for e in evs]) * 100
    half = lambda f: [e for e in evs if f(e["year"])]  # noqa: E731
    early, late = half(lambda y: y <= 2023), half(lambda y: y >= 2024)
    return {"n": len(evs), "win20": round(float((r20 > 0).mean() * 100), 1),
            "avg20": round(float(r20.mean()), 2), "median20": round(float(np.median(r20)), 2),
            "avg10": round(float(r10.mean()), 2), "excess20": round(float(r20.mean() - base), 2),
            "bounce": round(float(np.mean([e["bounce"] for e in evs]) * 100), 1),
            "break": round(float(np.mean([e["broke"] for e in evs]) * 100), 1),
            "mdd": round(float(np.mean([e["mdd"] for e in evs]) * 100), 1),
            "win20_early": round(float(np.mean([e["r20"] > 0 for e in early]) * 100), 1) if early else None,
            "win20_late": round(float(np.mean([e["r20"] > 0 for e in late]) * 100), 1) if late else None}


def main() -> None:
    import yfinance as yf
    names = {**US, **KR}
    agg: dict[str, list] = {}
    bases = []
    by_mkt: dict[str, dict[str, list]] = {"US": {}, "KR": {}}
    tickers = list(names)
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        raw = yf.download(chunk, period="6y", interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
        for t in chunk:
            try:
                df = (raw[t] if len(chunk) > 1 else raw)[["Open", "High", "Low", "Close", "Volume"]].dropna()
            except Exception:  # noqa: BLE001
                continue
            if len(df) < 400:
                continue
            c = df["Close"]
            base = float((c.shift(-H2) / c - 1).dropna().iloc[260:].mean() * 100)
            bases.append(base)
            mk = "US" if t in US else "KR"
            for k, evs in evaluate_ticker(df).items():
                for e in evs:
                    e["exc"] = e["r20"] * 100 - base
                agg.setdefault(k, []).extend(evs)
                by_mkt[mk].setdefault(k, []).extend(evs)
    base_all = float(np.mean(bases)) if bases else 0.0
    out = {"updated": datetime.now(KST).isoformat(timespec="minutes"), "tickers": len(bases),
           "baseline20": round(base_all, 2),
           "note": "미래 정보 없이(그 시점에 확정된 저점·거래량만) 계산. 진입=지지선+1% 지정가, 같은 종목·방법은 10일 간격.",
           "methods": {k: summarize(v, base_all) for k, v in agg.items()},
           "byMarket": {m: {k: summarize(v, base_all) for k, v in d.items()} for m, d in by_mkt.items()}}
    for k, v in agg.items():   # 종목별 평소 수익 대비 초과(종목 편향 제거)
        if v:
            out["methods"][k]["excessTicker20"] = round(float(np.mean([e["exc"] for e in v])), 2)
    (DATA / "sr_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["methods"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
