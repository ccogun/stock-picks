"""한국 종목 외국인·기관 순매매(수급). 네이버 금융 종목별 '외국인·기관 순매매' 표를 하루 한 번 받는다.

flows.json: {ticker: {"date": 최근 거래일, "f1","i1": 당일 순매매 수량, "f5","i5": 최근 5일 합계,
                      "v5": 5일 외국인+기관 순매수 대금 ÷ 20일 평균 거래대금 (수급 강도)}}
장중에는 전날까지 확정된 자료를 쓰므로 미래 정보를 미리 보는 일은 없다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd
import requests

from common import DATA, load_json, now_kst, save_json

FLOW_FILE = DATA / "flows.json"
LAST_ERR: list[str] = []
UA = {"User-Agent": "Mozilla/5.0 (stock-picks bot)", "Referer": "https://finance.naver.com/"}


def _num(s) -> float:
    try:
        return float(str(s).replace(",", "").replace("+", "").strip())
    except ValueError:
        return float("nan")


def _fetch(ticker: str) -> tuple[str, dict | None]:
    code = ticker.split(".")[0]
    try:
        r = requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}", headers=UA, timeout=10)
        r.encoding = "euc-kr"
        tables = pd.read_html(StringIO(r.text))
    except Exception as e:  # noqa: BLE001
        LAST_ERR.append(f"{ticker}: {type(e).__name__} {e}"[:300] + (f" status={r.status_code} len={len(r.text)}" if "r" in locals() else ""))
        return ticker, None
    LAST_ERR.append(f"{ticker}: tables={[ (t.shape, [str(c) for c in t.columns][:9]) for t in tables[:4]]}"[:900])
    for t in tables:
        if t.shape[1] < 7:
            continue
        flat = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if not any("기관" in c for c in flat) or not any("외국인" in c for c in flat):
            continue
        t.columns = flat
        t = t.dropna(how="all")
        date_col = flat[0]
        t = t[t[date_col].astype(str).str.match(r"\d{4}\.\d{2}\.\d{2}")]
        if len(t) < 5:
            continue
        inst_col = next(c for c in flat if "기관" in c and "순매매" in c)
        frgn_col = next(c for c in flat if "외국인" in c and "순매매" in c)
        close_col = next(c for c in flat if "종가" in c)
        t = t.head(20)
        inst, frgn, close = t[inst_col].map(_num), t[frgn_col].map(_num), t[close_col].map(_num)
        vol_col = next((c for c in flat if "거래량" in c), None)
        avg_val = float((t[vol_col].map(_num) * close).mean()) if vol_col else float("nan")
        net5 = float(((inst.head(5) + frgn.head(5)) * close.head(5)).sum())
        return ticker, {"date": str(t[date_col].iloc[0]).replace(".", "-"),
                        "f1": float(frgn.iloc[0]), "i1": float(inst.iloc[0]),
                        "f5": float(frgn.head(5).sum()), "i5": float(inst.head(5).sum()),
                        "v5": round(net5 / avg_val, 3) if avg_val and avg_val == avg_val else None}
    return ticker, None


def refresh(tickers: list[str], force: bool = False) -> dict:
    """하루 한 번(장 마감 후 또는 오늘 자료가 없을 때) 갱신."""
    cache = load_json(FLOW_FILE, {})
    now = now_kst()
    last = cache.get("_updated", "")
    after_close = now.hour * 60 + now.minute >= 15 * 60 + 40
    if force or not last:
        need = True
    else:
        from datetime import datetime, timedelta
        last_dt = datetime.fromisoformat(last)
        today_close = now.replace(hour=15, minute=40, second=0, microsecond=0)
        need = (after_close and last_dt < today_close) or (now - last_dt > timedelta(hours=20))
    if not need:
        return cache
    got = 0
    with ThreadPoolExecutor(6) as ex:
        for t, d in ex.map(_fetch, list(dict.fromkeys(tickers))):
            if d:
                cache[t] = d
                got += 1
    if got:
        cache["_updated"] = now.isoformat(timespec="minutes")
        save_json(FLOW_FILE, cache)
    print(f"[flows] {got}/{len(tickers)}종목 수급 갱신")
    if not got:   # 진단용: 실패 원인 샘플 저장
        save_json(DATA / "flows_debug.json", {"at": now.isoformat(timespec="minutes"), "samples": LAST_ERR[:5]})
    return cache


def flow_tag(f: dict | None) -> str:
    """'동반 순매수' | '한쪽 순매수' | '순매도' | '미확인' (최근 5일 기준)"""
    if not f:
        return "미확인"
    fb, ib = f.get("f5", 0) > 0, f.get("i5", 0) > 0
    if fb and ib:
        return "동반 순매수"
    if fb or ib:
        return "한쪽 순매수"
    return "순매도"
