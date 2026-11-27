"""가격 알림: docs/data/alerts.json 의 가격대에 닿았는지 매시간 확인.

alerts.json 형식:
{"alerts": [{"id": "soxl-1", "ticker": "SOXL", "name": "SOXL", "level": 30.0, "dir": "below"|"above",
             "note": "듬뿍아범 지지 가격대", "source": "...", "active": true,
             "hitAt": null, "hitPrice": null, "notified": false}]}
- 직전 확인 이후의 5분봉(프리·애프터 포함) 저가/고가로 판정 → 한 시간 사이에 잠깐 닿은 것도 잡는다.
- 닿으면 hitAt/hitPrice 기록, active=false. 알림 발송은 예약 작업(Claude)이 notified=false 인 항목을 보고 한다.
"""
from __future__ import annotations

import pandas as pd

from common import DATA, load_json, now_kst, save_json

FILE = DATA / "alerts.json"


def main() -> None:
    j = load_json(FILE, {"alerts": []})
    act = [a for a in j.get("alerts", []) if a.get("active", True) and not a.get("hitAt")]
    if not act:
        return
    import yfinance as yf
    now = now_kst()
    since = pd.Timestamp(j.get("checkedAt") or (now - pd.Timedelta(hours=2)).isoformat())
    tickers = sorted({a["ticker"] for a in act})
    raw = yf.download(tickers, period="2d", interval="5m", prepost=True, group_by="ticker",
                      auto_adjust=False, threads=False, progress=False)
    for t in tickers:
        try:
            df = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(subset=["Close"])
        except Exception as e:  # noqa: BLE001
            print(f"[alerts] {t} 시세 실패: {e}")
            continue
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        win = df[df.index > since.tz_convert(df.index.tz) - pd.Timedelta(minutes=10)]
        if win.empty:
            win = df.tail(1)
        last = float(df["Close"].iloc[-1])
        for a in act:
            if a["ticker"] != t:
                continue
            a["last"], a["lastAt"] = round(last, 4), df.index[-1].isoformat()
            lo, hi = float(win["Low"].min()), float(win["High"].max())
            hit = lo <= a["level"] if a.get("dir", "below") == "below" else hi >= a["level"]
            if hit:
                bar = win[win["Low"] <= a["level"]] if a.get("dir", "below") == "below" else win[win["High"] >= a["level"]]
                a.update(hitAt=bar.index[0].tz_convert("Asia/Seoul").isoformat(timespec="minutes"),
                         hitPrice=round(float(a["level"]), 4), active=False, notified=False)
                print(f"[alerts] 도달: {t} {a['level']} ({a.get('note', '')})")
    j["checkedAt"] = now.isoformat(timespec="minutes")
    save_json(FILE, j)


if __name__ == "__main__":
    main()
