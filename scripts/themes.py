"""이슈 테마 감시·추천 (한국).

- 네이버 금융 테마 순위(전 테마)를 읽어 오늘·3일 강한 테마를 고른다. 정치·선거·전쟁·계절·문화·질병·정책·
  산업 등 테마 종류에 제한이 없다(이름으로 분류만 한다).
- 예약 작업(Claude)이 쓰는 두 파일을 함께 쓴다.
  docs/data/theme_notes.json  {테마명: {"date","why","kind"}}  — 오늘 오른 이유(뉴스 재료)
  docs/data/theme_watch.json  [{"match","why","kind","until"}]  — 곧 이슈가 될 테마(선거일·계절·행사 등)
- 추천 규칙: 강한 테마(또는 예정 이슈 테마) 안에서 '아직 덜 오른' 후발주만. 상한가·급등 대장주 추격 금지.
  당일 +15% 미만, 대장주보다 5%p 이상 덜 오름, 20일선 위 상승 추세, RSI 75 미만, 거래대금 기준 통과,
  외국인·기관 동반 순매도 아님, 적자·고부채 아님. 하루 최대 theme_max 종목(학습으로 0~2 조정, 0이면 모의 기록).
결과: docs/data/themes.json
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import requests

from common import DATA, MARKETS, fmt_price, load_json, now_kst, save_json

THEMES_FILE = DATA / "themes.json"
NOTES_FILE = DATA / "theme_notes.json"
WATCH_FILE = DATA / "theme_watch.json"
KRX_FILE = DATA / "krx_list.json"
UA = {"User-Agent": "Mozilla/5.0 (stock-picks bot)", "Referer": "https://finance.naver.com/"}
BASE = "https://finance.naver.com"

KIND_WORDS = [
    ("정치·선거", "정치 선거 대선 총선 지방선거 대통령 남북 대북 개헌 통일 인맥"),
    ("전쟁·지정학", "방위 방산 전쟁 우크라이나 러시아 재건 이스라엘 중동 국방 군사 드론 사이버 보안 테러 북한"),
    ("계절·날씨", "겨울 여름 난방 에어컨 냉방 폭염 장마 황사 미세먼지 독감 감기 빙과 김장 제설 날씨 한파 가뭄 홍수 태풍 "
                 "비료 농업 사료 수산 수능"),
    ("질병·보건", "백신 코로나 전염병 마스크 치료제 진단 엠폭스 원숭이두창 바이러스 독감 비만"),
    ("문화·소비", "엔터 콘텐츠 드라마 영화 게임 웹툰 음원 여행 항공 카지노 화장품 면세 캐릭터 아이돌 스포츠 올림픽 "
                 "월드컵 의류 식품 음식료 주류 편의점 반려동물 K-"),
    ("정책·제도", "정부 정책 규제 원전 부동산 건설 출산 육아 지역화폐 탄소 밸류업 지배구조 공매도 금리 "
                 "증권 은행 보험 SOC 철도"),
]


def kind_of(name: str) -> str:
    for kind, words in KIND_WORDS:
        if any(w in name for w in words.split()):
            return kind
    return "산업·기술"


def _get(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=20)
    r.encoding = "euc-kr"
    return r.text


def _json(url: str):
    r = requests.get(url, headers={**UA, "Referer": "https://m.stock.naver.com/"}, timeout=20)
    r.raise_for_status()
    return r.json()


def _list_in(j) -> list[dict]:
    """응답에서 가장 큰 dict 목록을 찾는다(키 이름이 바뀌어도 동작하도록)."""
    if isinstance(j, list):
        return [x for x in j if isinstance(x, dict)]
    best = []
    if isinstance(j, dict):
        for v in j.values():
            cand = _list_in(v) if isinstance(v, (dict, list)) else []
            if len(cand) > len(best):
                best = cand
    return best


def _pick(d: dict, *keys, default=None):
    low = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low and low[k.lower()] not in (None, ""):
            return low[k.lower()]
    return default


def _f(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("%", "").replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def fetch_theme_list(max_pages: int = 5) -> list[dict]:
    """네이버 모바일 JSON: /api/stocks/theme?page=N&pageSize=100"""
    out, seen, sample = [], set(), None
    for page in range(1, max_pages + 1):
        items = _list_in(_json(f"https://m.stock.naver.com/api/stocks/theme?page={page}&pageSize=100"))
        sample = sample or (items[:2] if items else None)
        new = 0
        for it in items:
            no = str(_pick(it, "no", "themeNo", "groupNo", "id", "code", default=""))
            name = _pick(it, "name", "themeName", "groupName")
            if not no or not name or no in seen:
                continue
            seen.add(no)
            out.append({"no": no, "name": str(name).strip(),
                        "chg": _f(_pick(it, "changeRate", "fluctuationsRatio", "fluctuationRate", "rate", default=0)),
                        "chg3": _f(_pick(it, "changeRate3Days", "threeDaysChangeRate", "recent3DaysChangeRate",
                                         "threeDayRate", default=0)),
                        "up": int(_f(_pick(it, "riseCount", "upCount", "risingCount", "increaseCount", default=0))),
                        "flat": int(_f(_pick(it, "steadyCount", "unchangedCount", "flatCount", default=0))),
                        "down": int(_f(_pick(it, "fallCount", "downCount", "fallingCount", "decreaseCount", default=0)))})
            new += 1
        if not new or len(items) < 100:
            break
    save_json(DATA / "themes_debug.json", {"listSample": sample})
    return out


def fetch_theme_detail(no: str) -> dict:
    j = _json(f"https://m.stock.naver.com/api/stocks/theme/{no}?page=1&pageSize=100")
    items = _list_in(j)
    stocks = []
    for it in items:
        code = str(_pick(it, "itemCode", "stockCode", "code", "reutersCode", default=""))[:6]
        if not re.fullmatch(r"\d{6}", code):
            continue
        ex = _pick(it, "stockExchangeType", "stockExchangeName", "marketType", default="")
        ex = str(ex.get("nameEng") or ex.get("code") or ex.get("name") if isinstance(ex, dict) else ex).upper()
        mk = "KOSDAQ" if ("KOSDAQ" in ex or ex in ("KQ", "코스닥")) else ("KOSPI" if ("KOSPI" in ex or ex in ("KS", "코스피")) else None)
        stocks.append({"code": code, "name": str(_pick(it, "stockName", "itemName", "name", default=code)),
                       "chg": _f(_pick(it, "fluctuationsRatio", "changeRate", default=0)), "mk": mk,
                       "why": str(_pick(it, "themeReason", "reason", "description", "info", default=""))[:120]})
    info = j.get("groupInfo") or j.get("themeInfo") or {} if isinstance(j, dict) else {}
    desc = str(_pick(info, "description", "desc", "info", default="")) if isinstance(info, dict) else ""
    dbg = load_json(DATA / "themes_debug.json", {})
    if "detailSample" not in dbg:
        dbg["detailSample"] = {"keys": list(j.keys()) if isinstance(j, dict) else None, "items": items[:2]}
        save_json(DATA / "themes_debug.json", dbg)
    return {"stocks": stocks, "desc": desc[:200]}


def krx_markets() -> dict[str, str]:
    """종목코드 → 'KOSPI'|'KOSDAQ' (7일 캐시)"""
    c = load_json(KRX_FILE, {})
    if c.get("at") and now_kst() - datetime.fromisoformat(c["at"]) < timedelta(days=7) and len(c.get("m", {})) > 1000:
        return c["m"]
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")
        m = {}
        for code, mk in zip(df["Code"], df["Market"]):
            mk = str(mk).upper()
            if mk.startswith("KOSDAQ"):
                m[str(code)] = "KOSDAQ"
            elif mk == "KOSPI":
                m[str(code)] = "KOSPI"
        if len(m) > 1000:
            save_json(KRX_FILE, {"at": now_kst().isoformat(timespec="minutes"), "m": m})
            return m
    except Exception as e:  # noqa: BLE001
        print("[themes] KRX 목록 실패:", e)
    return c.get("m", {})


def _active_watch(now) -> list[dict]:
    today = now.strftime("%Y-%m-%d")
    return [w for w in load_json(WATCH_FILE, []) if isinstance(w, dict) and w.get("match")
            and (not w.get("until") or w["until"] >= today)]


def build(now) -> tuple[list[dict], dict]:
    """강한 테마 + 예정 이슈 테마 선정, 테마별 종목 목록까지. (themes, 종목코드→시장)"""
    allt = fetch_theme_list()
    if not allt:
        return [], {}
    notes = load_json(NOTES_FILE, {})
    watch = _active_watch(now)
    for t in allt:
        t["kind"] = (notes.get(t["name"]) or {}).get("kind") or kind_of(t["name"])
        t["score"] = round(t["chg"] + 0.5 * t["chg3"], 2)
        w = next((w for w in watch if w["match"] in t["name"]), None)
        if w:
            t["watch"] = {"why": w.get("why", ""), "until": w.get("until"), "kind": w.get("kind")}
            if w.get("kind"):
                t["kind"] = w["kind"]
    hot = [t for t in allt if t["chg"] >= 2.0 and (t["up"] + t["down"] == 0 or t["up"] >= max(2 * t["down"], 3))]
    hot = sorted(hot, key=lambda t: -t["score"])[:6]
    sel = hot + [t for t in allt if t.get("watch") and t not in hot][:6]
    with ThreadPoolExecutor(6) as ex:
        for t, d in zip(sel, ex.map(lambda t: fetch_theme_detail(t["no"]), sel)):
            t.update(d)
            t["hot"] = t in hot
            n = notes.get(t["name"]) or {}
            if n.get("why"):
                t["why"], t["whyDate"] = n["why"], n.get("date")
    return sel, krx_markets()


def run(picks: list[dict], now, idx_data: dict, mkt_info: dict, regular: bool, frac: float) -> list[str]:
    from fundamentals import refresh as refresh_fund
    from flows import flow_tag, refresh as refresh_flows
    from combined import unsafe
    from params import load_params
    from run import atr, download_kr, entry_from, market_date, rsi, write_prices

    try:
        sel, mk_of = build(now)
    except Exception as e:  # noqa: BLE001
        print("[themes] 테마 수집 실패:", e)
        return []
    tick = {}
    for t in sel:
        for s in t.get("stocks", []):
            m = mk_of.get(s["code"]) or s.get("mk")
            if m:
                s["market"] = m
                s["ticker"] = s["code"] + (".KS" if m == "KOSPI" else ".KQ")
                tick[s["ticker"]] = m
    data = download_kr(list(tick), "1y") if tick else {}
    for t in sel:
        for s in t.get("stocks", []):
            df = data.get(s.get("ticker"))
            if df is not None and len(df) > 2:
                s["chg"] = round((float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100, 2)
        st = sorted([s for s in t.get("stocks", []) if s.get("chg") is not None], key=lambda s: -s["chg"])
        t["top"] = [{k: s.get(k) for k in ("name", "ticker", "market", "chg")} for s in st[:6]]
        t["count"] = len(t.get("stocks", []))
        if st:
            t["leaders"] = [{"name": s["name"], "chg": s["chg"]} for s in st[:2]]

    P = load_params()
    theme_max = int(P.get("theme_max", 2))
    paper = theme_max <= 0
    limit = 1 if paper else theme_max
    added = []
    cands = []
    if regular and frac >= 0.08:
        flows = refresh_flows(list(tick))
        fund = refresh_fund(list(tick))
        for t in sel:
            lead = t["leaders"][0]["chg"] if t.get("leaders") else 0
            for s in t.get("stocks", []):
                df = data.get(s.get("ticker"))
                if df is None or len(df) < 60:
                    continue
                m = s["market"]
                c, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
                chg = (c / prev - 1) * 100
                ma20 = df["Close"].rolling(20).mean()
                r = float(rsi(df["Close"]).iloc[-1])
                avg_val = float((df["Close"] * df["Volume"]).iloc[-21:-1].mean())
                vr = float(df["Volume"].iloc[-1]) / (float(df["Volume"].iloc[-21:-1].mean()) * max(frac, 0.05) or 1)
                lo_chg = -3 if t.get("watch") and not t.get("hot") else 0
                if not (lo_chg < chg < 15 and chg <= lead - 5 and c > ma20.iloc[-1] > ma20.iloc[-6]
                        and r < 75 and avg_val >= MARKETS[m]["min_value"] * 0.5):
                    continue
                tag = flow_tag(flows.get(s["ticker"]))
                if tag == "동반 순매도" or unsafe(fund.get(s["ticker"])):
                    continue
                a = float(atr(df).iloc[-1])
                stop = min(max(c - 1.5 * a, float(ma20.iloc[-1]) * 0.98), c * 0.95)
                cands.append({"t": t, "s": s, "c": c, "chg": chg, "rsi": r, "vr": vr, "flow": tag, "stop": stop,
                              "target": c + 2 * (c - stop), "df": df,
                              "score": t["score"] * 2 + min(vr, 4) * 3 + (5 if tag == "동반 순매수" else 0)
                              + (6 if t.get("watch") else 0)})
        cands.sort(key=lambda x: -x["score"])
        notes = load_json(NOTES_FILE, {})
        active = {p["ticker"] for p in picks if p["status"] == "active"}
        used_theme = {(p.get("features") or {}).get("theme") for p in picks if p["method"] == "theme"
                      and p["recDate"] == market_date("KOSPI", now)}
        n = sum(1 for p in picks if p["method"] == "theme" and p["recDate"] == market_date("KOSPI", now))
        for x in cands:
            if n >= limit:
                break
            t, s, m = x["t"], x["s"], x["s"]["market"]
            if s["ticker"] in active or t["name"] in used_theme:
                continue
            cur = MARKETS[m]["currency"]
            ld = " · ".join(f"{l['name']} {l['chg']:+.1f}%" for l in t.get("leaders", []))
            why = (notes.get(t["name"]) or {}).get("why") or (t.get("watch") or {}).get("why") \
                or t.get("desc") or "재료 확인 전 — 예약 작업이 뉴스로 보강"
            if s.get("why"):
                why += f" / 편입 사유: {s['why']}"
            today = market_date(m, now)
            idx = idx_data.get(MARKETS[m]["index"])
            p = {"id": f"{today}-{m}-theme-{s['ticker']}", "market": m, "method": "theme",
                 "ticker": s["ticker"], "name": s["name"], "currency": cur,
                 "recAt": now.isoformat(timespec="minutes"), "recDate": today, "recPrice": round(x["c"], 2),
                 "recIndex": float(idx["Close"].iloc[-1]) if idx is not None else None,
                 "reasons": [
                     f"테마: {t['name']}({t['kind']}) 오늘 {t['chg']:+.1f}% · 3일 {t['chg3']:+.1f}% · 상승 {t['up']}/하락 {t['down']}"
                     + (f" · 대장주 {ld}" if ld else "") + (" · 예정 이슈 테마" if t.get("watch") else ""),
                     f"재료: {why}"[:220],
                     f"종목: 오늘 {x['chg']:+.1f}% — 대장주보다 덜 오른 후발주 · 거래량 평소의 {x['vr']:.1f}배(장중 보정)",
                     f"차트·수급: 20일선 위 상승 추세, RSI {x['rsi']:.0f} · 외국인·기관 5일 {x['flow']}",
                     f"리스크: 테마주는 재료가 꺼지면 급락 — 손절 {fmt_price(x['stop'], cur)} ({(x['stop'] / x['c'] - 1) * 100:.1f}%), "
                     f"목표 {fmt_price(x['target'], cur)} ({(x['target'] / x['c'] - 1) * 100:+.1f}%), 최대 5거래일 보유",
                 ],
                 "source": "이슈 테마 후발주" + (" (모의 — 성적 부진으로 일시 중단 중)" if paper else ""),
                 "features": {"signal": "theme", "theme": t["name"], "category": t["kind"],
                              "watch": bool(t.get("watch")), "flow": x["flow"], "rsi": round(x["rsi"], 1),
                              "regime": (mkt_info.get(m) or {}).get("regime")},
                 "status": "active", "verdict": "판단 중", "stop": round(x["stop"], 2),
                 "target": round(x["target"], 2), "paper": paper,
                 "chartNote": "이슈 테마 안에서 대장주보다 덜 오른 종목을 20일선(주황) 위 추세일 때 잡은 자리(▲)입니다. "
                              "테마주는 변동이 커서 손절선(빨간 점선)을 짧게 잡았고 5거래일이 지나면 종료합니다."}
            if (ef := entry_from(m, now)):
                p.update(entry="pending", entryFrom=ef, verdict="진입 대기")
            picks.append(p)
            write_prices(s["ticker"], x["df"])
            active.add(s["ticker"])
            used_theme.add(t["name"])
            n += 1
            added.append(f"{m}:{s['name']}(테마)")
            t.setdefault("picked", []).append(s["name"])

    for t in sel:
        t.pop("stocks", None)
    save_json(THEMES_FILE, {"updated": now.isoformat(timespec="minutes"), "mode": "모의" if paper else "실전",
                            "max": limit, "themes": sel,
                            "candidates": [{"name": x["s"]["name"], "theme": x["t"]["name"], "chg": round(x["chg"], 1),
                                            "score": round(x["score"], 1)} for x in cands[:10]]})
    print(f"[themes] 테마 {len(sel)}개, 후보 {len(cands)}종목, 신규 {added}")
    return added
