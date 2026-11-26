"""섹터별 주요 종목 지도(히트맵) 데이터. 한국(코스피·코스닥 상위)과 미국(S&P 500 상위)을 따로 만든다.

- 섹터·시가총액: 야후 파이낸스 종목 정보(일주일 캐시, 실행마다 일부씩 갱신)
- 등락률: 최근 두 거래일 종가(한국은 네이버·KRX 시세)
결과: docs/data/heatmap.json
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from common import DATA, KST, load_json, now_kst, save_json

META_FILE = DATA / "heatmap_meta.json"
OUT_FILE = DATA / "heatmap.json"
TOP = {"KR": 100, "US": 150}
PER_RUN = 200

SECTOR_KO = {
    "Technology": "기술", "Healthcare": "헬스케어", "Financial Services": "금융", "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재", "Communication Services": "커뮤니케이션", "Industrials": "산업재",
    "Energy": "에너지", "Utilities": "유틸리티", "Real Estate": "부동산", "Basic Materials": "소재",
}


def _meta(t: str) -> tuple[str, dict | None]:
    import yfinance as yf
    try:
        info = yf.Ticker(t).info or {}
        cap = info.get("marketCap")
        if not cap:
            return t, None
        return t, {"sector": info.get("sector") or "기타", "cap": float(cap),
                   "at": now_kst().isoformat(timespec="minutes")}
    except Exception:  # noqa: BLE001
        return t, None


def refresh_meta(tickers: list[str]) -> dict:
    meta = load_json(META_FILE, {})
    now = now_kst()
    stale = [t for t in tickers if t not in meta or
             now - datetime.fromisoformat(meta[t]["at"]).astimezone(KST) > timedelta(days=7)][:PER_RUN]
    if stale:
        with ThreadPoolExecutor(6) as ex:
            for t, d in ex.map(_meta, stale):
                if d:
                    meta[t] = d
        save_json(META_FILE, meta)
        print(f"[heatmap] 종목 정보 {len(stale)}개 갱신, 캐시 {len(meta)}개")
    return meta


def build(region: str, pool: list[list[str]], meta: dict, download) -> dict | None:
    names = dict((t, n) for t, n in pool)
    ranked = sorted([t for t in names if t in meta], key=lambda t: -meta[t]["cap"])[:TOP[region]]
    if len(ranked) < 10:
        return None
    px = download(ranked, period="5d")
    sectors: dict[str, list] = {}
    asof = None
    for t in ranked:
        df = px.get(t)
        if df is None or len(df) < 2:
            continue
        c0, c1 = float(df["Close"].iloc[-2]), float(df["Close"].iloc[-1])
        asof = max(asof or "", df.index[-1].strftime("%Y-%m-%d"))
        sec = SECTOR_KO.get(meta[t]["sector"], meta[t]["sector"] if meta[t]["sector"] != "기타" else "기타")
        sectors.setdefault(sec, []).append({"t": t, "n": names[t], "cap": round(meta[t]["cap"]),
                                            "chg": round((c1 / c0 - 1) * 100, 2), "px": round(c1, 2)})
    out = [{"name": s, "cap": sum(x["cap"] for x in v), "stocks": sorted(v, key=lambda x: -x["cap"])}
           for s, v in sectors.items()]
    out.sort(key=lambda s: -s["cap"])
    return {"asof": asof, "sectors": out}


def main() -> None:
    from run import download
    from universe import get_universe
    uni = get_universe()
    kr_pool = uni.get("KOSPI", []) + uni.get("KOSDAQ", [])
    us_pool = uni.get("SP500", [])
    # 한국 목록은 시가총액 순이라 앞쪽부터, 미국은 전체를 돌며 정보 채움
    meta = refresh_meta([t for t, _ in kr_pool[:160]] + [t for t, _ in us_pool])
    res = load_json(OUT_FILE, {})
    for region, pool in (("KR", kr_pool), ("US", us_pool)):
        r = build(region, pool, meta, download)
        if r:
            res[region] = r
    res["updated"] = now_kst().isoformat(timespec="minutes")
    save_json(OUT_FILE, res)
    print("[heatmap] 완료:", {k: len(v["sectors"]) for k, v in res.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
