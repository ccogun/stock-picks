"""종합 추천: 투자자 근거 + 차트(기술적) 근거가 함께 있는 종목.

build_candidates  — 한 시장의 기술적 신호(차트·전통전략 중 기술 신호·스윙)를 종목별로 묶고
                    안전장치(적자·고부채 제외)를 적용해 후보 목록을 만든다 → candidates.json
auto_combined     — 한국: 외국인·기관 5일 동반 순매수(투자자 근거)인 후보를 자동으로 종합 추천.
                    미국/그 밖의 투자자 근거(내부자 매수·13D·액티브 ETF 등)는 예약 작업(Claude)이
                    후보 목록에서 확인해 inbox 로 기록한다.
"""
from common import MARKETS, MAX_NEW_PER_DAY, fmt_price

TECH_STRATEGY = {"zone", "trendline", "reversal", "macd", "bollinger"}   # 재무형(value·dividend·quality) 제외
TREND = {"breakout", "pullback", "ext", "macd", "swing"}
LABEL = {"breakout": "20일 고점 돌파", "pullback": "20일선 눌림 반등", "ext": "시간외 갭 돌파",
         "swing": "전일 고가 돌파(단기)", "zone": "지지 가격대 반등", "trendline": "빗각 지지선 반등",
         "reversal": "RSI 과매도 반등", "macd": "MACD 상승 전환", "bollinger": "볼린저 하단 반등"}
MAX_DE = 3.0   # 부채비율 300%


def unsafe(f: dict | None) -> str | None:
    """적자·고부채면 사유 반환 (데이터 없으면 통과)"""
    if not f:
        return None
    if f.get("roe") is not None and f["roe"] < 0:
        return "적자(ROE 음수)"
    if f.get("de") is not None and f["de"] > MAX_DE:
        return f"고부채(부채비율 {f['de'] * 100:.0f}%)"
    return None


def build_candidates(market: str, tech: dict, fund: dict, flows: dict, reg: dict, data: dict,
                     reason_fns: dict) -> list[dict]:
    from flows import flow_tag
    by: dict[str, dict] = {}

    def add(o, sig, src, price, why=None):
        t = o["ticker"]
        e = by.setdefault(t, {"ticker": t, "name": o["name"], "price": round(float(price), 4), "signals": [],
                              "why": [], "rs_pct": round(float(o.get("rs_pct", 50)), 1), "_src": []})
        if sig not in e["signals"]:
            e["signals"].append(sig)
            e["why"].append(why or LABEL[sig])
            e["_src"].append((src, o))
        e["rs_pct"] = max(e["rs_pct"], round(float(o.get("rs_pct", 50)), 1))

    for c in tech.get("chart", []):
        sig = "ext" if c.get("ext") else ("breakout" if c["breakout"] else "pullback")
        add(c, sig, "chart", c["last"])
    for o in tech.get("strategy", []):
        if o["signal"] in TECH_STRATEGY:
            add(o, o["signal"], "strategy", o["x"]["c"], f"{LABEL[o['signal']]} — {o['why']}")
        elif o["signal"].startswith("book_"):
            add(o, o["signal"], "strategy", o["x"]["c"], f"『차트의 기술』 {o['why']}")
    for o in tech.get("swing", []):
        add(o, "swing", "swing", o["x"]["c"])

    out = []
    for t, e in by.items():
        bad = unsafe(fund.get(t))
        if bad:
            continue
        f = flows.get(t) if flows else None
        e["flow"] = flow_tag(f) if flows else None
        if e["flow"] == "동반 순매도":
            continue
        if f:
            e["f5"], e["i5"] = f.get("f5", 0), f.get("i5", 0)
        src, o = e.pop("_src")[0]       # 손절·목표는 첫 신호(차트 → 전략 → 스윙 순) 기준
        _, extra = reason_fns[src](o, market, reg)
        e["stop"], e["target"], e["chartNote"] = extra["stop"], extra["target"], extra.get("chartNote", "")
        from booksig import TREND_RULES
        e["trend"] = sum(s in TREND or (s.startswith("book_") and s[5:] in TREND_RULES) for s in e["signals"])
        e["support"] = len(e["signals"]) - e["trend"]
        e["score"] = round(len(e["signals"]) * 10 + {"동반 순매수": 20, "한쪽 순매수": 5}.get(e["flow"], 0)
                           + e["rs_pct"] / 10, 1)
        out.append(e)
    return sorted(out, key=lambda e: -e["score"])


def _num(v: float) -> str:
    return f"{v:+,.0f}주"


def auto_combined(market: str, cands: list[dict], picks: list[dict], today: str, now, idx, reg: dict) -> list[dict]:
    cfg = MARKETS[market]
    if cfg["region"] != "KR":
        return []
    n = sum(1 for p in picks if p["market"] == market and p["method"] == "combined" and p["recDate"] == today)
    active = {p["ticker"] for p in picks if p["method"] == "combined" and p["status"] == "active"}
    new = []
    for e in cands:
        if n >= MAX_NEW_PER_DAY:
            break
        if e["flow"] != "동반 순매수" or e["ticker"] in active:
            continue
        cur, label, c = cfg["currency"], cfg["label"], e["price"]
        reasons = [
            f"투자자 근거: 최근 5일 외국인 {_num(e['f5'])} · 기관 {_num(e['i5'])} 동반 순매수",
            "차트 근거: " + " / ".join(e["why"]),
            f"겹친 근거: 투자자 1 + 차트 {len(e['signals'])}개 (추세 {e['trend']} · 지지 {e['support']}) · "
            f"{label} 감시 종목 중 상대강도 상위 {100 - e['rs_pct']:.0f}%",
            f"시장 국면: {label} {reg['regime']} — {reg['note']}",
            f"리스크 관리: 손절 {fmt_price(e['stop'], cur)} ({(e['stop'] / c - 1) * 100:.1f}%), 1차 목표 "
            f"{fmt_price(e['target'], cur)} ({(e['target'] / c - 1) * 100:+.1f}%)",
        ]
        new.append({"id": f"{today}-{market}-combined-{e['ticker']}", "market": market, "method": "combined",
                    "ticker": e["ticker"], "name": e["name"], "currency": cur,
                    "recAt": now.isoformat(timespec="minutes"), "recDate": today,
                    "recPrice": round(c, 4), "recIndex": float(idx["Close"].iloc[-1]),
                    "reasons": reasons, "source": "종합(수급 + 차트 신호 일치)",
                    "features": {"signal": "+".join(e["signals"]), "signals": e["signals"], "nSignals": len(e["signals"]),
                                 "regime": reg["regime"], "rsPct": e["rs_pct"], "flow": e["flow"], "style": "수급"},
                    "status": "active", "verdict": "판단 중",
                    "stop": e["stop"], "target": e["target"], "chartNote": e["chartNote"]})
        active.add(e["ticker"])
        n += 1
    return new
