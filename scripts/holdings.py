"""내가 산 종목 감시: docs/data/holdings.json(앱이 동기화)을 매시간 확인해 대응이 바뀌면 알림을 만든다.

holdings.json: {"updated": "...", "items": [{"id", "ticker", "name", "market", "currency", "price", "at"}]}
- 판단 규칙은 앱(index.html holdingPlan)과 같다: 손절(최대 -8%), 경계·절반 손절, 손절선 매수가로, 과열 1/3 익절,
  1차 목표 절반 익절, 2차 목표 1/4 익절, 추적 손절(최고 종가 - 2ATR).
- 이전과 '할 일'이 달라졌을 때만 alerts.json 에 kind="holding" 알림을 추가(notified=false) → 예약 작업이 푸시.
"""
from __future__ import annotations

import pandas as pd

from common import DATA, load_json, now_kst, save_json

HOLD = DATA / "holdings.json"
STATE = DATA / "holdings_state.json"
ALERTS = DATA / "alerts.json"
SHORT = {"swing", "theme", "theme_cand"}
QUIET = {"보유", "보유 · 추가 매수 금지", "보유 (추적 손절)"}   # 이 상태로 바뀔 때는 알리지 않음


def fmt(v: float, cur: str) -> str:
    return f"{v:,.0f}원" if cur == "KRW" else f"${v:,.2f}"


def plan(df: pd.DataFrame, cur: float, h: dict, pick: dict | None) -> tuple[str, str]:
    C, H, L = df["Close"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(C)
    b = float(h["price"])
    rp = float((pick or {}).get("recPrice") or b)
    k = b / rp
    tr = [max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1])) for i in range(n - 14, n)]
    atr = sum(tr) / len(tr)
    ma20 = C[n - 20:].mean()
    g = sum(max(C[i] - C[i - 1], 0) for i in range(n - 14, n))
    lo = sum(max(C[i - 1] - C[i], 0) for i in range(n - 14, n))
    rsi = 100 - 100 / (1 + g / lo) if lo else 100
    since = h.get("at") or (pick or {}).get("recDate") or str(df.index[-1].date())
    after = df[df.index >= pd.Timestamp(since)]
    hi_c = max([b, cur, *after["Close"].tolist()])
    stop = max(float(pick["stop"]) * k if pick and pick.get("stop") else b - 2 * atr, b * 0.92)
    t1 = max(float(pick["target"]) * k if pick and pick.get("target") else 0, b + 2 * (b - stop))
    t2 = b + (t1 - b) * 1.8
    R = max(b - stop, 1e-9)
    cu = h.get("currency", "KRW")
    P = lambda v: fmt(v, cu)                                         # noqa: E731
    pc = lambda v: f"{(v / b - 1) * 100:+.1f}%"                      # noqa: E731
    below20 = C[-1] < ma20 and C[-2] < ma20
    reached = hi_c >= t1
    trail = max(b, hi_c - 2 * atr)
    if cur <= stop:
        a, why = "전량 손절", f"손절선 {P(stop)}({pc(stop)}) 이탈"
    elif reached and cur <= trail:
        a, why = "남은 물량 정리", f"최고 종가 {P(hi_c)}에서 2ATR 넘게 하락(추적 손절선 {P(trail)})"
    elif cur >= t2:
        a, why = "1/4 추가 익절, 나머지는 추적", f"2차 목표 {P(t2)}({pc(t2)}) 도달, 추적 손절선 {P(trail)}"
    elif cur >= t1:
        a, why = "절반 익절", f"1차 목표 {P(t1)}({pc(t1)}) 도달 — 나머지는 손절선을 매수가로"
    elif cur > b and rsi > 75 and cur > ma20 + 2.5 * atr:
        a, why = "1/3 부분 익절", f"단기 과열(RSI {rsi:.0f}, 20일선 위 2.5ATR 이상)"
    elif reached:
        a, why = "보유 (추적 손절)", f"추적 손절선 {P(trail)}"
    elif cur - b >= R:
        a, why = "보유 · 손절선을 매수가로 올리기", f"손절 폭만큼 수익 — 손절선을 {P(b)}로"
    elif cur >= b:
        a, why = "보유", f"1차 익절 {P(t1)}, 손절 {P(stop)}"
    else:
        f = (b - cur) / R
        if f >= 0.5 and below20:
            a, why = "절반 부분 손절", f"손절선까지 절반 넘게 하락 + 20일선 아래 2일 마감, 나머지 손절 {P(stop)}"
        elif f >= 0.5:
            a, why = "경계 · 추가 매수 금지", f"손절선 {P(stop)}까지 {(cur / stop - 1) * 100:.1f}% 남음"
        else:
            a, why = "보유 · 추가 매수 금지", f"손절 {P(stop)}({pc(stop)})"
    if pick and pick.get("method") in SHORT and len(after) >= 5 and "손절" not in a and "정리" not in a:
        a += " · 보유 기한(5거래일) 지남"
    return a, f"현재 {P(cur)}({(cur / b - 1) * 100:+.1f}%) · {why}"


def main() -> None:
    items = load_json(HOLD, {}).get("items", [])
    if not items:
        return
    from run import download
    picks = {p["id"]: p for p in load_json(DATA / "picks.json", {}).get("picks", [])}
    data = download([h["ticker"] for h in items], "6mo")
    state = load_json(STATE, {})
    alerts = load_json(ALERTS, {"alerts": []})
    now = now_kst()
    new = 0
    for h in items:
        df = data.get(h["ticker"])
        if df is None or len(df) < 25 or not h.get("price"):
            continue
        pick = picks.get(h["id"])
        cur = float((pick or {}).get("lastPrice") or df["Close"].iloc[-1])
        if pick and pick.get("lastDate") and str(df.index[-1].date()) > pick["lastDate"]:
            cur = float(df["Close"].iloc[-1])
        action, detail = plan(df, cur, h, pick)
        key = f"{h['id']}|{h['price']}"
        prev = state.get(key, {}).get("action")
        state[key] = {"action": action, "detail": detail, "at": now.isoformat(timespec="minutes")}
        if action != prev and not (prev is None and action in QUIET) and action not in QUIET:
            alerts.setdefault("alerts", []).append({
                "id": f"hold-{h['id']}-{now.strftime('%m%d%H%M')}", "kind": "holding", "ticker": h["ticker"],
                "name": h.get("name", h["ticker"]), "msg": f"{h.get('name')} — {action}: {detail}",
                "active": False, "hitAt": now.isoformat(timespec="minutes"), "notified": False})
            new += 1
            print(f"[holdings] {h.get('name')}: {prev} → {action}")
    # 오래된 보유 알림 정리(알림 끝난 것 30개만 유지)
    hold_done = [a for a in alerts["alerts"] if a.get("kind") == "holding" and a.get("notified")]
    if len(hold_done) > 30:
        drop = {a["id"] for a in hold_done[:-30]}
        alerts["alerts"] = [a for a in alerts["alerts"] if a["id"] not in drop]
    save_json(STATE, state)
    save_json(ALERTS, alerts)
    print(f"[holdings] {len(items)}종목 확인, 새 알림 {new}")


if __name__ == "__main__":
    main()
