"""시장별 감시 종목(유니버스) 목록. 일주일에 한 번 새로 받고, 실패하면 이전 목록을 쓴다."""
from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

from common import DATA, KST, load_json, save_json

UNIVERSE_FILE = DATA / "universe.json"
UA = {"User-Agent": "Mozilla/5.0 (stock-picks bot; github actions)"}

# 목록을 못 받을 때 쓰는 최소 목록
FALLBACK = {
    "KOSPI": [["005930.KS", "삼성전자"], ["000660.KS", "SK하이닉스"], ["207940.KS", "삼성바이오로직스"],
              ["005380.KS", "현대차"], ["105560.KS", "KB금융"], ["015760.KS", "한국전력"],
              ["035420.KS", "NAVER"], ["012450.KS", "한화에어로스페이스"], ["068270.KS", "셀트리온"]],
    "KOSDAQ": [["196170.KQ", "알테오젠"], ["247540.KQ", "에코프로비엠"], ["086520.KQ", "에코프로"],
               ["141080.KQ", "리가켐바이오"], ["298380.KQ", "에이비엘바이오"], ["214450.KQ", "파마리서치"]],
    "SP500": [["AAPL", "Apple"], ["MSFT", "Microsoft"], ["AMZN", "Amazon"], ["JPM", "JPMorgan Chase"],
              ["XOM", "Exxon Mobil"], ["DHR", "Danaher"], ["KO", "Coca-Cola"], ["DIS", "Walt Disney"]],
    "NASDAQ": [["AAPL", "Apple"], ["MSFT", "Microsoft"], ["NVDA", "NVIDIA"], ["AMZN", "Amazon"],
               ["COST", "Costco"], ["PEP", "PepsiCo"], ["GILD", "Gilead"], ["AMAT", "Applied Materials"]],
}


def _wiki_tables(url: str) -> list[pd.DataFrame]:
    html = requests.get(url, headers=UA, timeout=30).text
    return pd.read_html(StringIO(html))


def _kr(market: str, top: int, suffix: str) -> list[list[str]]:
    import FinanceDataReader as fdr  # 설치는 requirements.txt
    df = fdr.StockListing(market)
    cap_col = "Marcap" if "Marcap" in df.columns else "MarCap"
    df = df.sort_values(cap_col, ascending=False)
    # 우선주·스팩·ETF 제외
    df = df[~df["Name"].str.contains("스팩|우$|우B$|우C$|리츠", regex=True, na=False)]
    return [[f"{c}{suffix}", n] for c, n in zip(df["Code"].head(top), df["Name"].head(top))]


def _sp500() -> list[list[str]]:
    t = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return [[str(s).replace(".", "-"), str(n)] for s, n in zip(t["Symbol"], t["Security"])]


def _nasdaq100() -> list[list[str]]:
    for t in _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
        cols = {c.lower(): c for c in t.columns.astype(str)}
        if "ticker" in cols and ("company" in cols or "security" in cols):
            name_col = cols.get("company") or cols.get("security")
            return [[str(s).replace(".", "-"), str(n)] for s, n in zip(t[cols["ticker"]], t[name_col])]
    raise RuntimeError("Nasdaq-100 table not found")


BUILDERS = {
    "KOSPI": lambda: _kr("KOSPI", 200, ".KS"),
    "KOSDAQ": lambda: _kr("KOSDAQ", 150, ".KQ"),
    "SP500": _sp500,
    "NASDAQ": _nasdaq100,
}


def get_universe(force: bool = False) -> dict[str, list[list[str]]]:
    cached = load_json(UNIVERSE_FILE, {})
    fresh = False
    if cached.get("updated"):
        age = datetime.now(KST) - datetime.fromisoformat(cached["updated"])
        fresh = age < timedelta(days=7)
    if fresh and not force:
        return cached["markets"]

    markets = {}
    for m, build in BUILDERS.items():
        try:
            lst = build()
            if len(lst) < 20:
                raise RuntimeError("too few")
            markets[m] = lst
        except Exception as e:  # noqa: BLE001
            print(f"[universe] {m} 목록 갱신 실패: {e}")
            markets[m] = cached.get("markets", {}).get(m) or FALLBACK[m]
    save_json(UNIVERSE_FILE, {"updated": datetime.now(KST).isoformat(timespec="seconds"),
                              "markets": markets})
    return markets


if __name__ == "__main__":
    u = get_universe(force=True)
    print({k: len(v) for k, v in u.items()})
