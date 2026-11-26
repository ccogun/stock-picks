"""『차트의 기술』(김정환) 매수 기법 — 검증(book_compare.py)과 실시간 스캔(strategies.py)이 같이 쓰는 신호 계산.

모든 신호는 t일 종가까지의 정보만 쓴다(미래 정보 없음). 반환: {기법: bool 배열}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL = {
    "granville1": "그랜빌 ①: 하락하던 20일선이 옆으로 눕고 주가가 상향 돌파",
    "granville2": "그랜빌 ②: 상승 중인 20일선 아래로 일시 이탈(조정 마무리)",
    "granville3": "그랜빌 ③: 상승 중인 20일선에 닿고 되오름(지지)",
    "granville4": "그랜빌 ④: 20일선 아래로 크게 벌어진 급락(회귀 매수)",
    "disparity20": "20일 이격도 95% 아래(단기 과매도)",
    "disparity60": "60일 이격도 90% 아래(중기 과매도)",
    "golden": "진짜 골든크로스: 상승 중인 60일선을 20일선이 상향 돌파 + 거래량 증가",
    "golden_fake": "가짜 골든크로스(하락 중인 60일선 돌파) — 비교용",
    "align": "정배열 전환: 주가 > 5일 > 20일 > 60일 > 120일선",
    "squeeze": "볼린저 스퀴즈 돌파: 밴드 폭이 좁아진 뒤 상단 돌파",
    "bb_reentry": "볼린저 하단 밖에서 밴드 안으로 재진입",
    "sar": "파라볼릭 SAR 매수 전환",
    "stoch": "스토캐스틱 슬로우 %D가 25를 아래에서 위로 돌파",
    "cci": "CCI가 -100을 아래에서 위로 돌파(과매도 탈출)",
    "cci0": "상승 추세에서 CCI 0선 상향 돌파",
    "rsi50": "상승 추세에서 RSI 50 상향 돌파",
    "dmi": "DMI: +DI가 -DI 돌파 후 돌파일 고가 돌파(EPR) + ADX 상승",
    "ichimoku": "일목균형표: 전환선 > 기준선 호전 + 구름대 위 + 후행스팬 호전",
    "double_bottom": "이중바닥 목선 돌파(거래량 증가)",
    "gap": "돌파갭: 옆으로 기던 주가가 거래량 급증과 함께 갭 상승하며 60일 신고가",
    "hammer": "하락 뒤 망치형 캔들",
    "engulfing": "하락 뒤 상승장악형 캔들",
    "piercing": "하락 뒤 관통형 캔들",
    "morning_star": "하락 뒤 샛별형 캔들",
}
TREND_RULES = {"granville1", "golden", "align", "squeeze", "sar", "cci0", "rsi50", "dmi", "ichimoku",
               "double_bottom", "gap"}


def _cross_up(a: pd.Series, b) -> np.ndarray:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return ((a > b) & (a.shift(1) <= b.shift(1))).to_numpy()


def _sar(h: np.ndarray, l: np.ndarray, af0=0.02, afmax=0.2) -> np.ndarray:
    """파라볼릭 SAR. 반환: +1(매수 구간) / -1(매도 구간)"""
    n = len(h)
    side = np.zeros(n)
    if n < 3:
        return side
    up, sar, ep, af = True, l[0], h[0], af0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + af0, afmax)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + af0, afmax)
        side[i] = 1 if up else -1
    return side


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    ma = {k: c.rolling(k).mean() for k in (5, 20, 60, 120)}
    m20, m60 = ma[20], ma[60]
    vavg = v.rolling(20).mean().shift(1)
    up20 = m20 > m20.shift(5)
    up60 = m60 > m60.shift(5)
    first = lambda cond: (cond & ~cond.shift(1, fill_value=False)).to_numpy()   # noqa: E731
    S: dict[str, np.ndarray] = {}

    # 그랜빌 매수 4법칙 (20일선)
    slope_past = m20.shift(5) / m20.shift(15) - 1
    slope_now = m20 / m20.shift(5) - 1
    S["granville1"] = _cross_up(c, m20) & (slope_past < -0.01).to_numpy() & (slope_now.abs() < 0.005).to_numpy()
    S["granville2"] = first((c < m20) & (c > m20 * 0.97) & up20 & (c.shift(1) >= m20.shift(1)))
    S["granville3"] = ((l <= m20 * 1.01) & (c > m20) & (c > o) & up20 & (c.shift(3) > m20.shift(3) * 1.02)).to_numpy()
    S["granville4"] = first((c / m20 < 0.88) & ~up20)

    # 이격도
    S["disparity20"] = first(c / m20 < 0.95)
    S["disparity60"] = first(c / m60 < 0.90)

    # 골든크로스(진짜/가짜), 정배열
    gc = _cross_up(m20, m60)
    S["golden"] = gc & up60.to_numpy() & (v > 1.3 * vavg).rolling(3).max().fillna(0).astype(bool).to_numpy()
    S["golden_fake"] = gc & ~up60.to_numpy()
    S["align"] = first((c > ma[5]) & (ma[5] > m20) & (m20 > m60) & (m60 > ma[120]))

    # 볼린저
    sd = c.rolling(20).std()
    upper, lower = m20 + 2 * sd, m20 - 2 * sd
    bw = (upper - lower) / m20
    narrow = bw.shift(1) <= bw.shift(1).rolling(120).quantile(0.2)
    S["squeeze"] = _cross_up(c, upper) & narrow.to_numpy()
    S["bb_reentry"] = ((c.shift(1) < lower.shift(1)) & (c > lower)).to_numpy()

    # 파라볼릭 SAR
    side = _sar(h.to_numpy(), l.to_numpy())
    S["sar"] = (side == 1) & (np.roll(side, 1) == -1)
    S["sar"][:2] = False

    # 스토캐스틱 슬로우(14,3,3)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    d = k.rolling(3).mean()
    S["stoch"] = _cross_up(d, 25)

    # CCI(20)
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    S["cci"] = _cross_up(cci, -100)
    S["cci0"] = _cross_up(cci, 0) & up60.to_numpy()

    # RSI 50
    dlt = c.diff()
    gain = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    S["rsi50"] = _cross_up(rsi, 50) & up60.to_numpy()

    # DMI/ADX + EPR(돌파일 고가 돌파)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    upm, dnm = h.diff(), -l.diff()
    pdm = upm.where((upm > dnm) & (upm > 0), 0.0)
    ndm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    xi = pd.Series(np.where(_cross_up(pdi, ndi), h, np.nan), index=c.index).ffill(limit=5)
    epr = (c > xi) & (c.shift(1) <= xi.shift(1)) & (pdi > ndi) & (adx > adx.shift(3))
    S["dmi"] = epr.to_numpy()

    # 일목균형표(9, 26, 52)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanA = ((tenkan + kijun) / 2).shift(26)
    spanB = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    ichi = (tenkan > kijun) & (c > cloud_top) & (c > c.shift(26))
    S["ichimoku"] = first(ichi)

    # 이중바닥 목선 돌파: 확정된 10일 스윙 저점 두 개(±3%, 10~60일 간격), 사이 고점 = 목선
    lo = l.to_numpy()
    hi = h.to_numpy()
    cc = c.to_numpy()
    vv, va = v.to_numpy(), vavg.to_numpy()
    db = np.zeros(n, bool)
    piv = [i for i in range(10, n - 10) if lo[i] == lo[i - 10:i + 11].min()]
    for t in range(80, n):
        ps = [p for p in piv if p + 10 <= t - 1 and p >= t - 80]
        if len(ps) < 2:
            continue
        a, b = ps[-2], ps[-1]
        if not (10 <= b - a <= 60 and abs(lo[b] / lo[a] - 1) <= 0.03):
            continue
        neck = hi[a:b + 1].max()
        if cc[t] > neck >= cc[t - 1] and vv[t] > 1.3 * (va[t] or np.inf):
            db[t] = True
    S["double_bottom"] = db

    # 돌파갭
    rng40 = h.shift(1).rolling(40).max() / l.shift(1).rolling(40).min() - 1
    S["gap"] = ((o > h.shift(1) * 1.01) & (l > h.shift(1)) & (c > h.shift(1).rolling(60).max())
                & (rng40 <= 0.25) & (v > 2 * vavg)).to_numpy()

    # 캔들 상승 반전(하락 뒤에만): 20일선 아래 + 5일 수익률 -5% 이하
    down = ((c.shift(1) < m20.shift(1)) & (c.shift(1) / c.shift(6) - 1 <= -0.05))
    body = (c - o).abs()
    rngd = (h - l).replace(0, np.nan)
    lower_sh = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_sh = h - pd.concat([o, c], axis=1).max(axis=1)
    S["hammer"] = (down & (lower_sh >= 2 * body) & (upper_sh <= 0.25 * rngd) & (body > 0)).to_numpy()
    po, pc = o.shift(1), c.shift(1)
    S["engulfing"] = (down & (pc < po) & (c > o) & (o <= pc) & (c >= po)).to_numpy()
    S["piercing"] = (down & (pc < po) & (o < l.shift(1)) & (c > (po + pc) / 2) & (c < po)).to_numpy()
    o2, c2 = o.shift(2), c.shift(2)
    small1 = (c.shift(1) - o.shift(1)).abs() <= 0.3 * (c2 - o2).abs()
    S["morning_star"] = (down.shift(1, fill_value=False) & (c2 < o2) & small1
                         & (pd.concat([o.shift(1), c.shift(1)], axis=1).max(axis=1) < c2)
                         & (c > o) & (c > (o2 + c2) / 2)).to_numpy()

    for k_ in S:
        S[k_] = np.nan_to_num(np.asarray(S[k_], dtype=float)).astype(bool)
    return S
