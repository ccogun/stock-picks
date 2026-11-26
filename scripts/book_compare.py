"""『차트의 기술』 매수 기법 검증 — 75종목(한국·미국 대형주) 6년, 미래 정보 없는 워크포워드.

신호는 t일 종가까지로 판단, 다음 날 시가에 매수했다고 보고 10·20거래일 뒤 종가로 성과를 잰다.
같은 종목·기법은 10거래일 간격. '평소'는 같은 종목의 아무 날이나 샀을 때 평균(종목 효과 제거).
채택 기준: 표본 60건↑, 20일 초과수익(종목 평소 대비) +1.0%p↑, 20일 승률이 평소보다 2%p↑,
          앞·뒤 기간(3년씩) 모두 초과수익 플러스 — 한 시기에만 통한 기법 배제.
결과: docs/data/book_compare.json  (adopted = 실시간 스캔에 켜는 기법)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from booksig import LABEL, TREND_RULES, signals  # noqa: E402
from patterns import DATA, KR, KST, US  # noqa: E402
from common import load_json  # noqa: E402

US_SET: set = set()

GAP = 10
MIN_N, MIN_EX, MIN_WIN = 60, 1.0, 2.0


def download(tickers: list[str]):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        raw = yf.download(chunk, period="6y", interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
        for t in chunk:
            try:
                df = (raw[t] if len(chunk) > 1 else raw)[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > 300:
                    out[t] = df
            except Exception as e:  # noqa: BLE001
                print(f"[book] {t} 실패: {e}")
    return out


def summarize(evs: list[dict], base_win: float, split) -> dict:
    if not evs:
        return {"n": 0}
    r10 = np.array([e["r10"] for e in evs])
    r20 = np.array([e["r20"] for e in evs])
    ex = np.array([e["ex20"] for e in evs])
    early = [e["ex20"] for e in evs if e["d"] < split]
    late = [e["ex20"] for e in evs if e["d"] >= split]
    return {"n": len(evs), "avg10": round(r10.mean() * 100, 2), "avg20": round(r20.mean() * 100, 2),
            "median20": round(float(np.median(r20)) * 100, 2), "win20": round((r20 > 0).mean() * 100, 1),
            "winEdge": round((r20 > 0).mean() * 100 - base_win, 1),
            "excess20": round(ex.mean() * 100, 2),
            "early": round(float(np.mean(early)) * 100, 2) if early else None,
            "late": round(float(np.mean(late)) * 100, 2) if late else None,
            "nEarly": len(early), "nLate": len(late),
            "worst20": round(float(np.percentile(r20, 10)) * 100, 1),
            **_h40(evs, split)}


def _h40(evs, split) -> dict:
    e = [x for x in evs if x.get("ex40") is not None]
    if not e:
        return {}
    ex = np.array([x["ex40"] for x in e])
    r = np.array([x["r40"] for x in e])
    early = [x["ex40"] for x in e if x["d"] < split]
    late = [x["ex40"] for x in e if x["d"] >= split]
    return {"excess40": round(ex.mean() * 100, 2), "win40": round((r > 0).mean() * 100, 1),
            "early40": round(float(np.mean(early)) * 100, 2) if early else None,
            "late40": round(float(np.mean(late)) * 100, 2) if late else None}


def main() -> None:
    # 검증 대상: 대형주 75종목 + 앱 감시 종목 전체(코스피200·코스닥150·S&P500·나스닥100)
    uni = load_json(DATA / "universe.json", {}).get("markets", {})
    us_set = set(US) | {t for m in ("SP500", "NASDAQ") for t, _ in uni.get(m, [])}
    kr_set = set(KR) | {t for m in ("KOSPI", "KOSDAQ") for t, _ in uni.get(m, [])}
    global US_SET
    US_SET = us_set
    data = download(sorted(us_set | kr_set))
    events: dict[str, list[dict]] = {}
    by_mkt: dict[str, dict[str, list]] = {"US": {}, "KR": {}}
    base_r20, base_win_n, base_n, dates = [], 0, 0, []
    for t, df in data.items():
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 300:
            continue
        S = signals(df)
        o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
        n = len(df)
        r10 = np.full(n, np.nan)
        r20 = np.full(n, np.nan)
        r10[:n - 11] = c[11:] / o[1:n - 10] - 1
        r20[:n - 21] = c[21:] / o[1:n - 20] - 1
        r40 = np.full(n, np.nan)
        r40[:n - 41] = c[41:] / o[1:n - 40] - 1
        valid = ~np.isnan(r20)
        valid[:130] = False                         # 지표가 안정된 뒤부터
        tb = float(np.nanmean(r20[valid]))          # 이 종목 평소 20일 수익
        tb40 = float(np.nanmean(r40[valid & ~np.isnan(r40)]))
        base_r20.append(tb)
        base_win_n += int((r20[valid] > 0).sum())
        base_n += int(valid.sum())
        idx = df.index
        dates += [idx[130], idx[-21]]
        mk = "US" if t in US_SET else "KR"
        for k, arr in S.items():
            last = -999
            for i in np.flatnonzero(arr & valid):
                if i - last < GAP:
                    continue
                last = i
                e = {"t": t, "d": idx[i].strftime("%Y-%m-%d"), "r10": float(r10[i]), "r20": float(r20[i]),
                     "ex20": float(r20[i] - tb),
                     "ex40": float(r40[i] - tb40) if not np.isnan(r40[i]) else None,
                     "r40": float(r40[i]) if not np.isnan(r40[i]) else None}
                events.setdefault(k, []).append(e)
                by_mkt[mk].setdefault(k, []).append(e)
    base_win = base_win_n / max(base_n, 1) * 100
    alld = sorted(e["d"] for v in events.values() for e in v)
    split = alld[len(alld) // 2]                   # 신호 날짜 기준 앞/뒤 절반
    methods = {k: {**summarize(v, base_win, split), "label": LABEL[k], "type": "추세" if k in TREND_RULES else "역추세"}
               for k, v in events.items()}
    for k in LABEL:
        methods.setdefault(k, {"n": 0, "label": LABEL[k]})
    adopted = []
    for k, m in methods.items():
        ok20 = (m.get("n", 0) >= MIN_N and m["excess20"] >= MIN_EX
                and m["winEdge"] >= MIN_WIN and (m["early"] or 0) > 0 and (m["late"] or 0) > 0)
        # 추세 추종 기법은 오닐식 8주 보유 취지에 맞게 40일 성과로도 인정
        ok40 = (k in TREND_RULES and m.get("n", 0) >= MIN_N and (m.get("excess40") or 0) >= 2 * MIN_EX
                and (m.get("early40") or 0) > 0 and (m.get("late40") or 0) > 0 and m["winEdge"] >= 0)
        ok = k != "golden_fake" and (ok20 or ok40)
        m["by"] = "20일" if ok20 else ("40일" if ok40 else None)
        m["adopted"] = bool(ok)
        if ok:
            adopted.append(k)
    out = {"updated": datetime.now(KST).isoformat(timespec="minutes"), "tickers": len(data),
           "baseWin20": round(base_win, 1), "baseAvg20": round(float(np.mean(base_r20)) * 100, 2), "split": split,
           "rule": f"표본 {MIN_N}건↑ · 종목 평소 대비 20일 초과수익 +{MIN_EX}%p↑ · 승률 +{MIN_WIN}%p↑ · 앞/뒤 3년 모두 플러스"
                   f" (추세 추종 기법은 40일 초과수익 +{2 * MIN_EX}%p↑·앞/뒤 플러스도 인정)",
           "adopted": sorted(adopted, key=lambda k: -methods[k]["excess20"]),
           "methods": dict(sorted(methods.items(), key=lambda kv: -(kv[1].get("excess20") or -99))),
           "byMarket": {mk: {k: summarize(v, base_win, split) for k, v in d.items()} for mk, d in by_mkt.items()}}
    (DATA / "book_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("평소 승률", round(base_win, 1), "채택:", out["adopted"])
    for k, m in out["methods"].items():
        print(f"{k:16s} n={m.get('n')} ex20={m.get('excess20')} win={m.get('win20')} edge={m.get('winEdge')} "
              f"early={m.get('early')} late={m.get('late')} ex40={m.get('excess40')} e40={m.get('early40')}/"
              f"{m.get('late40')} {'✔' + m['by'] if m.get('adopted') else ''}")


if __name__ == "__main__":
    main()
