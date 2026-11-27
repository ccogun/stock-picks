"""즉시 폰 알림(ntfy): alerts.json 에서 notified=false 인 항목을 ntfy 로 보내고 notified=true 로 표시.

폰에 ntfy 앱(무료)을 설치하고 아래 TOPIC 을 구독하면 알림이 온다.
※ 공개 저장소라 이 주제 이름은 누구나 볼 수 있다(보유 종목·매수가도 이미 공개). 바꾸려면 이 값과 앱 구독만 바꾸면 된다.
"""
from __future__ import annotations

import requests

from common import DATA, load_json, save_json

TOPIC = "sp-c2a50f24aa22f3fd"
FILE = DATA / "alerts.json"


def main() -> None:
    j = load_json(FILE, {"alerts": []})
    sent = 0
    for a in j.get("alerts", []):
        if not a.get("hitAt") or a.get("notified"):
            continue
        if a.get("kind") == "holding":
            title, msg, tags, pr = "📌 내 종목", a.get("msg", ""), ["pushpin"], 4
        else:
            title = f"🔔 {a.get('name')} {a.get('level')} 도달"
            msg, tags, pr = f"{a.get('hitAt', '')[5:16].replace('T', ' ')} · {a.get('note', '')}", ["bell"], 5
        try:
            r = requests.post("https://ntfy.sh/", json={"topic": TOPIC, "title": title, "message": msg,
                                                        "tags": tags, "priority": pr,
                                                        "click": "https://ccogun.github.io/stock-picks/"}, timeout=15)
            if r.ok:
                a["notified"], a["notifiedBy"] = True, "ntfy"
                sent += 1
            else:
                print("[notify] 실패", r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            print("[notify] 오류", e)
    if sent:
        save_json(FILE, j)
    print(f"[notify] {sent}건 발송")


if __name__ == "__main__":
    main()
