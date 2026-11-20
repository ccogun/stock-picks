"""추천 결과를 분석해 다음 추천 기준을 조정한다 (매 실행 후 호출).

- 성과 지표: 10거래일 시점 지수 대비 초과수익(없으면 종료 시 초과수익). 손절로 일찍 끝난 것은 종료 시 값.
- 조정 원칙: 표본 8개 이상일 때만, 한 번에 조금씩, 같은 기준은 7일에 한 번까지, 정해진 범위 안에서.
- 결과는 docs/data/review.json(통계)과 docs/data/params.json(기준값·변경 이력)에 남는다.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from common import DATA, KST, MARKETS, load_json, now_kst, save_json
from params import BOUNDS, DEFAULTS, load_state, save_state

PICKS_FILE = DATA / "picks.json"
REVIEW_FILE = DATA / "review.json"
MIN_N = 8
COOLDOWN_DAYS = 7
WINDOW = 30   # 최근 30건 기준


def outcome(p: dict) -> float | None:
    snap = p.get("snap") or {}
    key = "5" if p.get("method") == "swing" else "10"   # 단기 스윙은 5거래일 기준
    if key in snap:
        return snap[key]
    if p.get("status") == "closed":
        return p.get("outcome", p.get("excess"))
    return None


def stats(items: list[dict]) -> dict:
    vals = [o for p in items if (o := outcome(p)) is not None]
    if not vals:
        return {"n": 0, "open": len(items)}
    wins = sum(1 for v in vals if v > 0)
    return {"n": len(vals), "open": len(items) - len(vals), "hit": round(wins / len(vals) * 100, 1),
            "avg": round(sum(vals) / len(vals), 2), "best": max(vals), "worst": min(vals)}


def group(picks: list[dict], key) -> dict:
    out: dict[str, list] = {}
    for p in picks:
        k = key(p)
        if k:
            out.setdefault(k, []).append(p)
    return {k: stats(v) for k, v in out.items()}


def adjust(state: dict, key: str, new, reason: str, now: datetime, sub: str | None = None) -> None:
    vals = state.setdefault("values", {})
    last = state.setdefault("lastChange", {})
    name = f"{key}.{sub}" if sub else key
    if name in last and now - datetime.fromisoformat(last[name]) < timedelta(days=COOLDOWN_DAYS):
        return
    lo, hi = BOUNDS.get(key, (None, None))
    if lo is not None:
        new = max(lo, min(hi, new))
    if sub:
        cur = {**DEFAULTS[key], **vals.get(key, {})}
        old = cur[sub]
        if old == new:
            return
        cur[sub] = new
        vals[key] = cur
    else:
        old = vals.get(key, DEFAULTS[key])
        if old == new:
            return
        vals[key] = new
    last[name] = now.isoformat(timespec="minutes")
    state.setdefault("history", []).append({"at": now.isoformat(timespec="minutes"), "param": name,
                                            "old": old, "new": new, "reason": reason})
    print(f"[review] {name}: {old} → {new} ({reason})")


def main() -> None:
    now = now_kst()
    picks = load_json(PICKS_FILE, {"picks": []})["picks"]
    picks = sorted(picks, key=lambda p: p["recAt"])
    chart = [p for p in picks if p["method"] == "chart"]
    inv = [p for p in picks if p["method"] == "investor"]
    strat = [p for p in picks if p["method"] == "strategy"]
    swing = [p for p in picks if p["method"] == "swing"]
    tech = chart + strat + swing
    sig = lambda p: (p.get("features") or {}).get("signal")  # noqa: E731

    review = {
        "updated": now.isoformat(timespec="minutes"),
        "metric": "10거래일 시점 지수 대비 초과수익(%p), 손절 종료는 종료 시점 값",
        "overall": {"investor": stats(inv), "chart": stats(chart), "strategy": stats(strat), "swing": stats(swing)},
        "byMarket": {m: {k: stats([p for p in lst if p["market"] == m])
                         for k, lst in (("investor", inv), ("chart", chart), ("strategy", strat), ("swing", swing))}
                     for m in MARKETS},
        "bySignal": group(tech, sig),
        "byRegime": group(picks, lambda p: (p.get("features") or {}).get("regime")),
        "byStyle": group(inv, lambda p: (p.get("features") or {}).get("style")),
        "stopOuts": sum(1 for p in picks if p.get("closeReason") == "손절선 이탈"),
    }

    # ── 차트 신호별 기준 조정 ──
    state = load_state()
    vals = {**DEFAULTS, **state.get("values", {})}
    for s, knob, step_tight, step_loose in (("breakout", "breakout_vol", 0.25, -0.1),
                                            ("pullback", "pullback_vol", 0.2, -0.1),
                                            ("ext", "ext_gap_min", 0.01, -0.005),
                                            ("swing", "swing_vol", 0.25, -0.1),
                                            ("value", None, 0, 0), ("dividend", None, 0, 0),
                                            ("quality", None, 0, 0), ("reversal", None, 0, 0),
                                            ("macd", None, 0, 0), ("bollinger", None, 0, 0)):
        st = stats([p for p in tech if sig(p) == s][-WINDOW:])
        if st["n"] < MIN_N:
            continue
        w = {**DEFAULTS["weight"], **vals.get("weight", {})}[s]
        if st["hit"] < 40 and st["avg"] < 0:
            why = f"{s} 신호 최근 {st['n']}건 적중 {st['hit']}%, 평균 {st['avg']}%p → 기준 강화"
            if knob:
                adjust(state, knob, round(vals[knob] + step_tight, 3), why, now)
            adjust(state, "weight", w - 5, why, now, sub=s)
        elif st["hit"] > 60 and st["avg"] > 1:
            why = f"{s} 신호 최근 {st['n']}건 적중 {st['hit']}%, 평균 {st['avg']}%p → 기준 완화"
            if knob:
                adjust(state, knob, round(vals[knob] + step_loose, 3), why, now)
            adjust(state, "weight", w + 5, why, now, sub=s)

    # ── 시장 국면별 상대강도 하한 조정 ──
    for regime_name, knob in (("하락", "rs_min_bear"), ("상승", "rs_min"), ("조정", "rs_min")):
        st = stats([p for p in chart if (p.get("features") or {}).get("regime") == regime_name][-WINDOW:])
        if st["n"] < MIN_N:
            continue
        if st["hit"] < 40 and st["avg"] < 0:
            adjust(state, knob, vals[knob] + 5, f"{regime_name}장 차트 추천 적중 {st['hit']}% → 상대강도 기준 상향", now)
        elif st["hit"] > 60 and st["avg"] > 1:
            adjust(state, knob, vals[knob] - 2, f"{regime_name}장 차트 추천 적중 {st['hit']}% → 상대강도 기준 완화", now)

    # ── 손절 폭 조정 (투자자 방식) ──
    closed_inv = [p for p in inv if p.get("status") == "closed"][-WINDOW:]
    if len(closed_inv) >= MIN_N:
        stopped = [p for p in closed_inv if p.get("closeReason") == "손절선 이탈"]
        rate = len(stopped) / len(closed_inv)
        # 손절 후 다시 회복한 경우가 많으면 손절이 너무 촘촘한 것
        if rate > 0.5:
            adjust(state, "investor_stop", round(min(vals["investor_stop"] + 0.01, 0.12), 3),
                   f"투자자 방식 손절 비율 {rate:.0%} → 손절 폭 확대", now)
        elif rate < 0.15 and stats(closed_inv).get("avg", 0) < 0:
            adjust(state, "investor_stop", round(max(vals["investor_stop"] - 0.01, 0.05), 3),
                   f"손실 종료가 많지만 손절은 드묾 → 손절 폭 축소", now)

    review["params"] = {**DEFAULTS, **state.get("values", {})}
    review["recentChanges"] = state.get("history", [])[-10:]
    save_state(state)
    save_json(REVIEW_FILE, review)
    print("[review] 완료:", review["overall"])


if __name__ == "__main__":
    main()
