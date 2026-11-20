"""재무 지표 캐시 (PER·PBR·배당수익률·ROE·부채비율 등). 일주일에 한 번씩, 실행마다 일부 종목만 갱신한다."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from common import DATA, KST, load_json, now_kst, save_json

FUND_FILE = DATA / "fundamentals.json"
STALE_DAYS = 7
PER_RUN = 100


def _num(v):
    try:
        v = float(v)
        return v if v == v else None   # NaN 제외
    except (TypeError, ValueError):
        return None


def _fetch(ticker: str) -> tuple[str, dict | None]:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001
        return ticker, None
    dy = _num(info.get("trailingAnnualDividendYield"))
    if dy is None:
        dy = _num(info.get("dividendYield"))
        if dy is not None and dy > 1:   # 일부 버전은 % 단위
            dy /= 100
    de = _num(info.get("debtToEquity"))
    return ticker, {
        "per": _num(info.get("trailingPE")), "fper": _num(info.get("forwardPE")),
        "pbr": _num(info.get("priceToBook")), "dy": dy,
        "roe": _num(info.get("returnOnEquity")), "de": de / 100 if de is not None else None,
        "payout": _num(info.get("payoutRatio")), "om": _num(info.get("operatingMargins")),
        "sector": info.get("sector") or info.get("industry"),
        "at": now_kst().isoformat(timespec="minutes"),
    }


def refresh(tickers: list[str]) -> dict[str, dict]:
    cache = load_json(FUND_FILE, {})
    now = now_kst()

    def stale(t):
        x = cache.get(t)
        return not x or now - datetime.fromisoformat(x["at"]).astimezone(KST) > timedelta(days=STALE_DAYS)

    todo = [t for t in dict.fromkeys(tickers) if stale(t)][:PER_RUN]
    if todo:
        with ThreadPoolExecutor(4) as ex:
            for t, d in ex.map(_fetch, todo):
                if d:
                    cache[t] = d
        save_json(FUND_FILE, cache)
        print(f"[fundamentals] {len(todo)}종목 갱신, 캐시 {len(cache)}종목")
    return cache
