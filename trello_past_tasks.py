#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()
BASE = "https://api.trello.com/1"

def require_env():
    if not os.getenv("TRELLO_KEY") or not os.getenv("TRELLO_TOKEN"):
        print("ERROR: TRELLO_KEY と TRELLO_TOKEN を環境変数に設定してください。", file=sys.stderr)
        sys.exit(1)

def api_get(path: str, params: Dict[str, Any] | None = None, max_retries: int = 3):
    if params is None:
        params = {}
    params = {**params, "key": os.getenv("TRELLO_KEY"), "token": os.getenv("TRELLO_TOKEN")}
    url = f"{BASE}/{path}"
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=30)
        # レート制限 or 一時エラーの簡易リトライ
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            sleep_s = int(r.headers.get("Retry-After", "0")) or (2 * (attempt + 1))
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()

def list_my_boards(include_closed: bool) -> List[Dict[str, Any]]:
    boards = api_get("members/me/boards", {"fields": "id,name,closed"})
    if not include_closed:
        boards = [b for b in boards if not b.get("closed")]
    return boards

def list_archived_cards_on_board(board_id: str, since_date: str = None) -> List[Dict[str, Any]]:
    # アーカイブ済みのみ（＝完了扱い）
    params = {"fields": "id,name,dateLastActivity"}
    if since_date:
        params["since"] = since_date
    cards = api_get(f"boards/{board_id}/cards/closed", params)
    
    # since_dateが指定されている場合、さらにフィルタリング
    if since_date:
        since_dt = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
        filtered_cards = []
        for card in cards:
            if card.get("dateLastActivity"):
                card_dt = datetime.fromisoformat(card["dateLastActivity"].replace('Z', '+00:00'))
                if card_dt >= since_dt:
                    filtered_cards.append(card)
        return filtered_cards
    
    return cards

def build_done_map(
    board_id: str | None,
    include_closed_boards: bool,
    use_board_id_as_key: bool,
    since_date: str = None
) -> Dict[str, List[str]]:
    done_map: Dict[str, List[str]] = {}

    if board_id:
        board = api_get(f"boards/{board_id}", {"fields": "id,name,closed"})
        cards = list_archived_cards_on_board(board["id"], since_date)
        key = board["id"] if use_board_id_as_key else board["name"]
        done_map[key] = [c["name"] for c in cards]
        return done_map

    # 全ボード横断
    for b in list_my_boards(include_closed_boards):
        cards = list_archived_cards_on_board(b["id"], since_date)
        key = b["id"] if use_board_id_as_key else b["name"]
        done_map[key] = [c["name"] for c in cards]
    return done_map

def main():
    parser = argparse.ArgumentParser(
        description="Trello: ボード → 完了（アーカイブ済み）カードのタイトル配列マップをJSONで出力"
    )
    parser.add_argument("--board-id", help="単一ボードのみ対象（指定がなければ全ボード）")
    parser.add_argument("--include-closed-boards", action="store_true", help="閉じたボードも含める")
    parser.add_argument("--use-board-id-as-key", action="store_true", help="マップのキーをボード名ではなくボードIDにする")
    parser.add_argument("--show-counts", action="store_true", help="合計件数をstderrに表示")
    parser.add_argument("--since-date", default="2025-02-01T00:00:00Z", help="指定日時以降のタスクのみ対象（ISO 8601形式、デフォルト: 2025-02-01T00:00:00Z）")
    args = parser.parse_args()

    require_env()

    done_map = build_done_map(
        board_id=args.board_id,
        include_closed_boards=args.include_closed_boards,
        use_board_id_as_key=args.use_board_id_as_key,
        since_date=args.since_date
    )

    # JSON（標準出力）
    print(json.dumps(done_map, ensure_ascii=False, indent=2))

    if args.show_counts:
        total = sum(len(titles) for titles in done_map.values())
        print(f"完了（アーカイブ済み）カード合計: {total} 件", file=sys.stderr)
        print("\n各ボードでの完了タスク数:", file=sys.stderr)
        for board_name, titles in done_map.items():
            print(f"  {board_name}: {len(titles)}件", file=sys.stderr)

if __name__ == "__main__":
    main()
