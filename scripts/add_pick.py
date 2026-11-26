"""Claude(투자자 방식 분석)가 추천을 넣을 때 쓰는 도구.

사용법: python scripts/add_pick.py input.json
입력 JSON: [{"market": "KOSPI", "ticker": "015760.KS", "name": "한국전력",
            "reasons": ["…5줄…"], "source": "따라 한 투자자/근거",
            "recPrice": 30150 (선택), "stop": 27500 (선택), "target": 36000 (선택),
            "method": "combined", "style": "내부자매수"}]
method "combined"(종합) 은 docs/data/candidates.json 에 오늘 올라온 후보 종목만 가능하다.

추천은 docs/data/inbox/ 에 새 파일로 저장되고, GitHub Actions(run.py)가 picks.json에 합치면서
가격·차트·성과를 채운다. picks.json을 직접 고치지 않으므로 자동 갱신과 충돌하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import DATA, MARKETS, load_json, now_kst, save_json


def main(path: str) -> None:
    items = load_json(Path(path), [])
    if isinstance(items, dict):
        items = [items]
    now = now_kst()
    ok = []
    for it in items:
        if it.get("market") not in MARKETS:
            raise SystemExit(f"알 수 없는 시장: {it.get('market')}")
        for k in ("ticker", "name", "reasons"):
            if not it.get(k):
                raise SystemExit(f"{k} 누락: {it}")
        if it.get("method") == "combined":
            cm = (load_json(DATA / "candidates.json", {}).get("markets", {}).get(it["market"]) or {}).get("items", [])
            if it["ticker"] not in {c["ticker"] for c in cm}:
                raise SystemExit(f"{it['name']}: 오늘 차트 후보(candidates.json)에 없는 종목은 종합 추천 불가")
        if len(it["reasons"]) != 5:
            raise SystemExit(f"{it['name']}: 추천 사유는 정확히 5줄이어야 합니다")
        it["recAt"] = now.isoformat(timespec="minutes")
        ok.append(it)
    out = DATA / "inbox" / f"{now.strftime('%Y%m%d-%H%M%S')}.json"
    save_json(out, ok)
    print(f"저장: {out.relative_to(DATA.parent.parent)} ({len(ok)}건)")


if __name__ == "__main__":
    main(sys.argv[1])
