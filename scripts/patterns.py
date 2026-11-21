"""대형주 '가격대 규칙성' 탐색 — 반복해서 반등이 나온 지지 가격대를 찾고, 그 가격대에서 샀을 때 성과를 검증한다.

예) 'BE는 160달러 근처에서 사면 수익이 난 경우가 많았다' 같은 패턴을 데이터로 확인.

방법 (최근 3년 일봉, 분할 반영·배당 미반영 가격):
1. 지지 후보: 좌우 10거래일 중 최저가인 날(스윙 저점)을 모은다.
2. 스윙 저점들을 ±3% 안에서 묶어 '가격대'를 만든다. 저점이 3번 이상 모인 가격대만 남긴다.
3. 진입 이벤트: 직전 5거래일 저가가 모두 가격대 위(+3%)에 있다가 처음 가격대 상단에 닿은 날.
   진입가 = 그날 종가와 가격대 상단 중 낮은 값(지정가 매수 가정).
4. 성과: 진입 후 10·20거래일 수익률, 20거래일 내 최대 낙폭, 가격대 이탈률(20일 내 종가가 하단 -5% 아래).
5. 통과 기준: 이벤트 4회 이상, 20일 승률 70% 이상, 평균 20일 수익률이 같은 종목의 아무 날 평균보다 +3%p 이상.

주의: 과거 패턴이다. 지지선은 하락장·악재에서 깨질 수 있고(이탈률 참고), 표본이 작다.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
KST = timezone(timedelta(hours=9))

US = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan",
    "LLY": "Eli Lilly", "V": "Visa", "MA": "Mastercard", "UNH": "UnitedHealth", "XOM": "Exxon Mobil",
    "COST": "Costco", "WMT": "Walmart", "HD": "Home Depot", "PG": "Procter & Gamble", "JNJ": "Johnson & Johnson",
    "NFLX": "Netflix", "ORCL": "Oracle", "AMD": "AMD", "CRM": "Salesforce", "ADBE": "Adobe",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "MRK": "Merck", "ABBV": "AbbVie", "CVX": "Chevron",
    "BAC": "Bank of America", "GS": "Goldman Sachs", "CAT": "Caterpillar", "GE": "GE Aerospace", "MU": "Micron",
    "QCOM": "Qualcomm", "INTC": "Intel", "PLTR": "Palantir", "UBER": "Uber", "ISRG": "Intuitive Surgical",
    "NOW": "ServiceNow", "TSM": "TSMC ADR", "ASML": "ASML", "BE": "Bloom Energy", "VST": "Vistra",
    "CEG": "Constellation Energy", "GEV": "GE Vernova", "ANET": "Arista Networks", "COIN": "Coinbase",
    "SMCI": "Super Micro", "DIS": "Disney", "NKE": "Nike", "MCD": "McDonald's", "SBUX": "Starbucks",
}
KR = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "373220.KS": "LG에너지솔루션", "207940.KS": "삼성바이오로직스",
    "005380.KS": "현대차", "000270.KS": "기아", "068270.KS": "셀트리온", "105560.KS": "KB금융",
    "055550.KS": "신한지주", "035420.KS": "NAVER", "035720.KS": "카카오", "005490.KS": "POSCO홀딩스",
    "012450.KS": "한화에어로스페이스", "329180.KS": "HD현대중공업", "034020.KS": "두산에너빌리티",
    "015760.KS": "한국전력", "051910.KS": "LG화학", "006400.KS": "삼성SDI", "028260.KS": "삼성물산",
    "196170.KQ": "알테오젠", "247540.KQ": "에코프로비엠",
}

PIV = 10          # 스윙 저점 좌우 기간
BAND = 0.03       # 가격대 폭 ±3%
MIN_PIVOTS = 3
MIN_EVENTS = 4
H1, H2 = 10, 20


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        raw = yf.download(chunk, period="3y", interval="1d", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df[["Open", "High", "Low", "Close"]].dropna()
                if len(df) > 250:
                    out[t] = df
            except Exception as e:  # noqa: BLE001
                print(f"[patterns] {t} 실패: {e}")
    return out


def zones(df: pd.DataFrame) -> list[dict]:
    low = df["Low"].to_numpy()
    n = len(low)
    piv = [low[i] for i in range(PIV, n - PIV) if low[i] == low[i - PIV:i + PIV + 1].min()]
    piv.sort()
    groups: list[list[float]] = []
    for p in piv:
        if groups and p <= np.mean(groups[-1]) * (1 + 2 * BAND):
            groups[-1].append(p)
        else:
            groups.append([p])
    return [{"center": float(np.median(g)), "pivots": len(g)} for g in groups if len(g) >= MIN_PIVOTS]


def evaluate(df: pd.DataFrame, z: dict) -> dict | None:
    c = z["center"]; up = c * (1 + BAND); lo = c * (1 - BAND)
    low = df["Low"].to_numpy(); close = df["Close"].to_numpy(); idx = df.index
    ev = []
    i = 5
    while i < len(df) - H2:
        if low[i] <= up and low[i - 5:i].min() > up:
            entry = min(close[i], up)
            r10 = close[i + H1] / entry - 1
            r20 = close[i + H2] / entry - 1
            mdd = low[i + 1:i + H2 + 1].min() / entry - 1
            broke = bool((close[i + 1:i + H2 + 1] < lo * 0.95).any())
            ev.append({"date": idx[i].strftime("%Y-%m-%d"), "entry": round(float(entry), 2),
                       "ret10": round(r10 * 100, 1), "ret20": round(r20 * 100, 1),
                       "mdd": round(mdd * 100, 1), "broke": broke})
            i += H2  # 같은 하락에서 중복 집계 방지
        else:
            i += 1
    if len(ev) < MIN_EVENTS:
        return None
    r20 = np.array([e["ret20"] for e in ev])
    base = (pd.Series(close).shift(-H2) / pd.Series(close) - 1).dropna().mean() * 100
    win = float((r20 > 0).mean() * 100)
    avg = float(r20.mean())
    if win < 70 or avg - base < 3:
        return None
    last = float(close[-1])
    return {"center": round(c, 2), "low": round(lo, 2), "high": round(up, 2), "pivots": z["pivots"],
            "events": len(ev), "winRate": round(win), "avgRet20": round(avg, 1),
            "avgRet10": round(float(np.mean([e["ret10"] for e in ev])), 1),
            "baseline20": round(float(base), 1), "avgMdd": round(float(np.mean([e["mdd"] for e in ev])), 1),
            "breakRate": round(float(np.mean([e["broke"] for e in ev]) * 100)),
            "lastClose": round(last, 2), "distPct": round((last / c - 1) * 100, 1), "history": ev[-6:]}


def main() -> None:
    names = {**US, **KR}
    data = download(list(names))
    items = []
    for t, df in data.items():
        res = [r for r in (evaluate(df, z) for z in zones(df)) if r]
        # 현재가와 가까운 가격대(현재가가 가격대 -15%~+40%)를 우선, 그 안에서 승률·수익 순
        near = [r for r in res if -15 <= r["distPct"] <= 40]
        pool = near or res
        best = max(pool, key=lambda r: (r["winRate"], r["avgRet20"])) if pool else None
        if best:
            mk = "US" if t in US else "KR"
            items.append({"ticker": t, "name": names[t], "market": mk,
                          "currency": "USD" if mk == "US" else "KRW", **best,
                          "near": -5 <= best["distPct"] <= 8})
    items.sort(key=lambda x: (not x["near"], abs(x["distPct"])))
    out = {"updated": datetime.now(KST).isoformat(timespec="minutes"),
           "method": "최근 3년 스윙 저점이 3번 이상 모인 ±3% 가격대에, 위에서 처음 닿았을 때 매수 가정. "
                     "이벤트 4회 이상·20일 승률 70% 이상·평균 수익이 평소보다 +3%p 이상인 것만.",
           "scanned": len(data), "items": items}
    (DATA / "patterns.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[patterns] {len(data)}종목 스캔, {len(items)}개 패턴")


if __name__ == "__main__":
    main()
