"""공통 설정과 유틸리티."""
from __future__ import annotations

import json
import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
PRICES = DATA / "prices"
KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")

# 시장별 설정
MARKETS = {
    "KOSPI": {"label": "코스피", "index": "^KS11", "region": "KR", "currency": "KRW",
               "min_value": 5e9},      # 20일 평균 거래대금 50억원 이상
    "KOSDAQ": {"label": "코스닥", "index": "^KQ11", "region": "KR", "currency": "KRW",
                "min_value": 3e9},     # 30억원 이상
    "SP500": {"label": "S&P 500", "index": "^GSPC", "region": "US", "currency": "USD",
               "min_value": 5e7},      # $50M 이상
    "NASDAQ": {"label": "나스닥", "index": "^IXIC", "region": "US", "currency": "USD",
                "min_value": 5e7},
}

MAX_NEW_PER_DAY = 3          # 시장·방식별 하루 최대 신규 추천 수
HOLD_DAYS = 20               # 평가 기간(거래일)
GOOD_EXCESS = 0.03           # 지수 대비 +3%p 이상이면 '잘한 추천'
BAD_EXCESS = -0.05           # 지수 대비 -5%p 이하면 '아쉬운 추천'


def now_kst() -> datetime:
    return datetime.now(KST)


def session_state(region: str, at: datetime | None = None) -> tuple[bool, float]:
    """(장중 여부, 장 경과 비율 0~1)을 돌려준다."""
    at = at or datetime.now(KST)
    if region == "KR":
        local = at.astimezone(KST)
        start, end = time(9, 0), time(15, 30)
    else:
        local = at.astimezone(NY)
        start, end = time(9, 30), time(16, 0)
    if local.weekday() >= 5:
        return False, 1.0
    t = local.time()
    if t < start:
        return False, 1.0
    if t > end:
        return False, 1.0
    total = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
    done = (t.hour * 60 + t.minute) - (start.hour * 60 + start.minute)
    return True, max(done / total, 0.05)


def region_recent(region: str, at: datetime | None = None, grace_min: int = 90) -> bool:
    """장중이거나 장 마감 후 grace_min분 이내면 True (마감 직후 한 번 더 분석)."""
    at = at or datetime.now(KST)
    open_, _ = session_state(region, at)
    if open_:
        return True
    if region == "KR":
        local = at.astimezone(KST)
        close_m = 15 * 60 + 30
    else:
        local = at.astimezone(NY)
        close_m = 16 * 60
    if local.weekday() >= 5:
        return False
    m = local.hour * 60 + local.minute
    return close_m <= m <= close_m + grace_min


def load_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, path)


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("/", "_")


def fmt_price(v: float, currency: str) -> str:
    if currency == "KRW":
        return f"{v:,.0f}원"
    return f"${v:,.2f}"
