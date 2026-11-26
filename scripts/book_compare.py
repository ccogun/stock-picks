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
            "worst20": round(float(np.percentile(r20, 10)) * 100, 1)}


def main() -> None:
    names = {**US, **KR}
    data = download(list(names))
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
        valid = ~np.isnan(r20)
        valid[:130] = False                         # 지표가 안정된 뒤부터
        tb = float(np.nanmean(r20[valid]))          # 이 종목 평소 20일 수익
        base_r20.append(tb)
        base_win_n += int((r20[valid] > 0).sum())
        base_n += int(valid.sum())
        idx = df.index
        dates += [idx[130], idx[-21]]
        mk = "US" if t in US else "KR"
        for k, arr in S.items():
            last = -999
            for i in np.flatnonzero(arr & valid):
                if i - last < GAP:
                    continue
                last = i
                e = {"t": t, "d": idx[i].strftime("%Y-%m-%d"), "r10": float(r10[i]), "r20": float(r20[i]),
                     "ex20": float(r20[i] - tb)}
                events.setdefault(k, []).append(e)
                by_mkt[mk].setdefault(k, []).append(e)
    base_win = base_win_n / max(base_n, 1) * 100
    split = sorted(d.strftime("%Y-%m-%d") for d in dates)[len(dates) // 2]
    methods = {k: {**summarize(v, base_win, split), "label": LABEL[k], "type": "추세" if k in TREND_RULES else "역추세"}
               for k, v in events.items()}
    for k in LABEL:
        methods.setdefault(k, {"n": 0, "label": LABEL[k]})
    adopted = []
    for k, m in methods.items():
        ok = (k != "golden_fake" and m.get("n", 0) >= MIN_N and m["excess20"] >= MIN_EX
              and m["winEdge"] >= MIN_WIN and (m["early"] or 0) > 0 and (m["late"] or 0) > 0)
        m["adopted"] = bool(ok)
        if ok:
            adopted.append(k)
    out = {"updated": datetime.now(KST).isoformat(timespec="minutes"), "tickers": len(data),
           "baseWin20": round(base_win, 1), "baseAvg20": round(float(np.mean(base_r20)) * 100, 2), "split": split,
           "rule": f"표본 {MIN_N}건↑ · 종목 평소 대비 20일 초과수익 +{MIN_EX}%p↑ · 승률 +{MIN_WIN}%p↑ · 앞/뒤 3년 모두 플러스",
           "adopted": sorted(adopted, key=lambda k: -methods[k]["excess20"]),
           "methods": dict(sorted(methods.items(), key=lambda kv: -(kv[1].get("excess20") or -99))),
           "byMarket": {mk: {k: summarize(v, base_win, split) for k, v in d.items()} for mk, d in by_mkt.items()}}
    (DATA / "book_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("평소 승률", round(base_win, 1), "채택:", out["adopted"])
    for k, m in out["methods"].items():
        print(f"{k:14s} n={m.get('n')} ex20={m.get('excess20')} win={m.get('win20')} edge={m.get('winEdge')} "
              f"early={m.get('early')} late={m.get('late')} {'✔' if m.get('adopted') else ''}")


if __name__ == "__main__":
    main()
