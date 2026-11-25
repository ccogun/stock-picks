"""대형주 지지 가격대 — 미래 정보 없이 검증한 방식(sr_compare.py 비교 결과 반영).

방법 (최근 6년 일봉, 분할 반영 가격):
1. 지지 후보: 좌우 20거래일 중 최저가인 날(스윙 저점). 오른쪽 20일이 지나야 '확정'된다.
2. 확정된 스윙 저점들을 ±3% 안에서 묶어 2번 이상 모인 곳을 '가격대'로 본다.
3. 과거 사례는 그 날짜에 이미 가격대가 존재했던 경우만 센다(미래 저점 사용 금지).
4. 사례: 직전 5거래일 저가가 가격대 +3% 위에 있다가 처음 가격대에 닿은 날.
   - 닿은 날 매수(지정가 상단) 성과와, 다음 날 반등 확인(종가가 가격대 위 + 전날보다 상승) 후 매수 성과를 함께 기록.
5. 워크포워드 비교에서 '반등 확인 후 매수'가 가장 나았다(sr_compare.json). 그래서 현재 상태를
   '도달 — 반등 확인 대기' / '반등 확인 — 매수 후보'로 표시한다.

주의: 과거 통계다. 가격대별 표본이 적고, 하락장·악재에서는 지지선이 깨질 수 있다.
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

# 사용자가 지목한 종목: 기준 통과 여부와 관계없이 모든 가격대 성과를 따로 기록
WATCH = {"BE": {"note": "사용자 관찰: 160달러 근처 매수 시 수익이 많았음", "level": 160}}

PIV = 20          # 스윙 저점 좌우 기간 (비교 검증에서 가장 나음)
BAND = 0.03       # 가격대 폭 ±3%
MIN_PIVOTS = 2
MIN_EVENTS = 4
H1, H2 = 10, 20


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        raw = yf.download(chunk, period="6y", interval="1d", group_by="ticker",
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


def pivots(df: pd.DataFrame) -> list[tuple[int, float]]:
    low = df["Low"].to_numpy()
    n = len(low)
    return [(i, float(low[i])) for i in range(PIV, n - PIV) if low[i] == low[i - PIV:i + PIV + 1].min()]


def zones(piv: list[tuple[int, float]], upto: int) -> list[dict]:
    """upto 시점까지 확정된(i+PIV < upto) 저점으로 만든 가격대."""
    vals = sorted(v for i, v in piv if i + PIV < upto)
    groups: list[list[float]] = []
    for v in vals:
        if groups and v <= np.mean(groups[-1]) * (1 + 2 * BAND):
            groups[-1].append(v)
        else:
            groups.append([v])
    return [{"center": float(np.median(g)), "pivots": len(g)} for g in groups if len(g) >= MIN_PIVOTS]


def existed(piv, center: float, t: int) -> bool:
    """t 시점에 이 가격대(±3%)에 확정 저점이 2개 이상 있었나."""
    return sum(1 for i, v in piv if i + PIV < t and abs(v / center - 1) <= BAND) >= MIN_PIVOTS


def evaluate(df: pd.DataFrame, piv, z: dict) -> dict:
    c = z["center"]; up = c * (1 + BAND); lo = c * (1 - BAND)
    low, close, idx = df["Low"].to_numpy(), df["Close"].to_numpy(), df.index
    n = len(df)
    ev = []
    i = 5
    while i < n:
        if low[i] <= up and low[i - 5:i].min() > up and existed(piv, c, i):
            e = {"date": idx[i].strftime("%Y-%m-%d"), "entry": round(float(min(close[i], up)), 2)}
            if i + H2 < n:
                e["ret20"] = round((close[i + H2] / min(close[i], up) - 1) * 100, 1)
                e["broke"] = bool((close[i + 1:i + H2 + 1] < lo * 0.95).any())
            if i + 1 < n:
                conf = close[i + 1] >= up * 0.99 and close[i + 1] > close[i]
                e["confirmed"] = bool(conf)
                if conf and i + 1 + H2 < n:
                    e["cret20"] = round((close[i + 1 + H2] / close[i + 1] - 1) * 100, 1)
            ev.append(e)
            i += H2
        else:
            i += 1
    done = [e for e in ev if "ret20" in e]
    conf = [e for e in ev if "cret20" in e]
    base = float((pd.Series(close).shift(-H2) / pd.Series(close) - 1).dropna().mean() * 100)
    last = float(close[-1])
    # 현재 상태
    status = None
    if n >= 2 and low[-1] <= up and low[-6:-1].min() > up:
        status = "도달 — 내일 반등 확인 필요"
    elif n >= 3 and low[-2] <= up and low[-7:-2].min() > up and close[-1] >= up * 0.99 and close[-1] > close[-2]:
        status = "반등 확인 — 매수 후보"
    elif lo * 0.95 <= last < lo:
        status = "가격대 하단 이탈 중"
    r = lambda xs, k: round(float(np.mean([e[k] for e in xs])), 1) if xs else None  # noqa: E731
    w = lambda xs, k: round(float(np.mean([e[k] > 0 for e in xs]) * 100)) if xs else None  # noqa: E731
    return {"center": round(c, 2), "low": round(lo, 2), "high": round(up, 2), "pivots": z["pivots"],
            "events": len(done), "winRate": w(done, "ret20"), "avgRet20": r(done, "ret20"),
            "confEvents": len(conf), "confWinRate": w(conf, "cret20"), "confAvgRet20": r(conf, "cret20"),
            "baseline20": round(base, 1),
            "breakRate": round(float(np.mean([e["broke"] for e in done]) * 100)) if done else None,
            "lastClose": round(last, 2), "distPct": round((last / c - 1) * 100, 1),
            "status": status, "history": ev[-6:]}


def main() -> None:
    names = {**US, **KR}
    data = download(list(names))
    items = []
    for t, df in data.items():
        piv = pivots(df)
        res = [evaluate(df, piv, z) for z in zones(piv, len(df))]
        # 현재가 아래쪽(최대 -20%) 또는 바로 위(+5%)의 가격대만, 가까운 순 2개 — 성적으로 고르지 않음(선택 편향 방지)
        near = sorted([r for r in res if -5 <= -r["distPct"] <= 20 or r["status"]], key=lambda r: abs(r["distPct"]))[:2]
        mk = "US" if t in US else "KR"
        for r in near:
            items.append({"ticker": t, "name": names[t], "market": mk, "currency": "USD" if mk == "US" else "KRW",
                          **r, "near": -3 <= r["distPct"] <= 8})
    watch = []
    for t, w in WATCH.items():
        df = data.get(t)
        if df is None:
            continue
        piv = pivots(df)
        cand = zones(piv, len(df)) + ([{"center": float(w["level"]), "pivots": 0}] if w.get("level") else [])
        zs = [evaluate(df, piv, z) for z in cand]
        watch.append({"ticker": t, "name": names.get(t, t), "note": w["note"], "level": w.get("level"),
                      "zones": [z for z in zs if z["events"] or z["status"] or abs(z["distPct"]) < 30]})
    order = {"반등 확인 — 매수 후보": 0, "도달 — 내일 반등 확인 필요": 1, "가격대 하단 이탈 중": 3}
    items.sort(key=lambda x: (order.get(x["status"], 2), not x["near"], abs(x["distPct"])))
    summary = {}
    try:
        sc = json.loads((DATA / "sr_compare.json").read_text(encoding="utf-8"))
        summary = {k: sc["methods"].get(k) for k in ("piv20", "piv20_confirm", "piv10", "dip")}
        summary["updated"] = sc.get("updated")
    except Exception:  # noqa: BLE001
        pass
    out = {"updated": datetime.now(KST).isoformat(timespec="minutes"),
           "method": "좌우 20거래일 스윙 저점이 2번 이상 모인 ±3% 가격대. 과거 사례는 그 시점에 가격대가 이미 있던 경우만 집계(미래 정보 없음).",
           "scanned": len(data), "items": items, "watch": watch, "validation": summary}
    (DATA / "patterns.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[patterns] {len(data)}종목 스캔, {len(items)}개 가격대")


if __name__ == "__main__":
    main()
