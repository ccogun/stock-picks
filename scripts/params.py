"""추천 기준값. review.py가 성과를 보고 조금씩 조정하고, run.py가 읽어서 쓴다."""
from __future__ import annotations

import copy

from common import DATA, load_json, save_json

PARAMS_FILE = DATA / "params.json"

DEFAULTS = {
    "rs_min": 70,             # 상대강도 백분위 하한(보통·상승장)
    "rs_min_bear": 85,        # 하락장
    "breakout_vol": 1.5,      # 돌파 신호 거래량 배수
    "pullback_vol": 1.0,      # 눌림 반등 거래량 배수
    "pullback_rsi_lo": 45,
    "pullback_rsi_hi": 62,
    "rsi_max": 78,            # 과열 제외
    "ext_gap_min": 0.02,      # 미국 시간외 갭 하한
    "ext_gap_max": 0.15,
    "weight": {"breakout": 10, "pullback": 0, "ext": 0,      # 신호별 가산점(순위)
               "value": 0, "dividend": 0, "quality": 0, "reversal": 0, "macd": 0, "bollinger": 0, "swing": 0, "trendline": 0, "zone": 0},
    # 전통 전략
    "value_per_max_kr": 12, "value_per_max_us": 18, "value_pbr_max_kr": 1.0, "value_pbr_max_us": 3.0,
    "div_min_kr": 0.04, "div_min_us": 0.03, "quality_roe_min": 0.15, "reversal_rsi": 30,
    # 단기 스윙
    "swing_vol": 2.0,
    # 수급: 외국인·기관 5일 동반 순매수 종목 가산점(한국)
    "flow_bonus": 10,
    "pullback_in_bear": False,
    "investor_stop": 0.08,    # 투자자 방식 기본 손절·목표
    "investor_target": 0.15,
    "theme_max": 2,           # 이슈 테마 하루 최대 추천 수 (0 = 일시 중단, 모의 기록만)
}

# 조정 가능한 범위
BOUNDS = {
    "rs_min": (60, 90), "rs_min_bear": (75, 95), "breakout_vol": (1.2, 3.0), "pullback_vol": (0.8, 2.0),
    "ext_gap_min": (0.01, 0.05), "weight": (-20, 20), "swing_vol": (1.5, 4.0), "flow_bonus": (0, 30), "theme_max": (0, 2),
}


def load_params() -> dict:
    saved = load_json(PARAMS_FILE, {})
    p = copy.deepcopy(DEFAULTS)
    for k, v in saved.get("values", {}).items():
        if k in p:
            p[k] = {**p[k], **v} if isinstance(p[k], dict) else v
    return p


def load_state() -> dict:
    return load_json(PARAMS_FILE, {"values": {}, "history": [], "lastChange": {}})


def save_state(state: dict) -> None:
    state["history"] = state.get("history", [])[-100:]
    save_json(PARAMS_FILE, state)
