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


# 거래 가능 시간(정규장 + 시간외). 한국: 넥스트레이드 08:00~20:00, 미국: 프리 04:00 ~ 애프터 20:00(동부)
EXTENDED = {"KR": (time(8, 0), time(20, 0)), "US": (time(4, 0), time(20, 0))}


def session_phase(region: str, at: datetime | None = None) -> str:
    """'regular' | 'pre' | 'after' | 'closed'"""
    at = at or datetime.now(KST)
    local = at.astimezone(KST if region == "KR" else NY)
    if local.weekday() >= 5:
        return "closed"
    open_, _ = session_state(region, at)
    if open_:
        return "regular"
    s, e = EXTENDED[region]
    t = local.time()
    reg_start = time(9, 0) if region == "KR" else time(9, 30)
    if s <= t < reg_start:
        return "pre"
    if t <= e:
        return "after" if t > reg_start else "closed"
    return "closed"


def region_recent(region: str, at: datetime | None = None) -> bool:
    """정규장 또는 시간외 거래 시간이면 True."""
    return session_phase(region, at) != "closed"


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
