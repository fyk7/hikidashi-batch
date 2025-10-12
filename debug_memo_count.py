#!/usr/bin/env python3
"""
デバッグ用: メモ数とチャンク処理状況を確認するスクリプト
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from utils import parse_db_url

# Load environment variables
load_dotenv()

def main():
    db_config = parse_db_url(os.getenv('DATABASE_URL'))

    with psycopg2.connect(**db_config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            # 全メモ数
            cur.execute("SELECT COUNT(*) as total FROM memos")
            total_memos = cur.fetchone()['total']
            print(f"全メモ数: {total_memos}")

            # 削除されていないメモ数
            cur.execute("SELECT COUNT(*) as count FROM memos WHERE is_deleted = FALSE")
            not_deleted = cur.fetchone()['count']
            print(f"削除されていないメモ数: {not_deleted}")

            # ドラフトでないメモ数
            cur.execute("SELECT COUNT(*) as count FROM memos WHERE is_deleted = FALSE AND is_draft = FALSE")
            not_draft = cur.fetchone()['count']
            print(f"削除されておらず、ドラフトでないメモ数: {not_draft}")

            # チャンクが存在するメモ数
            cur.execute("""
                SELECT COUNT(DISTINCT memo_id) as count
                FROM memo_chunks
            """)
            with_chunks = cur.fetchone()['count']
            print(f"チャンクが存在するメモ数: {with_chunks}")

            # チャンクが存在しないメモ数（現在のget_unprocessed_memosのロジック）
            cur.execute("""
                SELECT COUNT(*) as count
                FROM memos m
                LEFT JOIN memo_chunks mc ON m.id = mc.memo_id
                WHERE m.is_deleted = FALSE
                  AND m.is_draft = FALSE
                  AND mc.id IS NULL
            """)
            unprocessed = cur.fetchone()['count']
            print(f"チャンクが存在しないメモ数: {unprocessed}")

            # チャンクが存在しないメモのIDを取得
            cur.execute("""
                SELECT m.id, m.title, LENGTH(m.content) as content_length,
                       m.created_at, m.updated_at
                FROM memos m
                LEFT JOIN memo_chunks mc ON m.id = mc.memo_id
                WHERE m.is_deleted = FALSE
                  AND m.is_draft = FALSE
                  AND mc.id IS NULL
                ORDER BY m.created_at DESC
                LIMIT 10
            """)
            unprocessed_memos = cur.fetchall()

            print(f"\nチャンクが存在しないメモの例（最大10件）:")
            for memo in unprocessed_memos:
                print(f"  ID: {memo['id']}, タイトル: {memo['title'][:50]}..., "
                      f"コンテンツ長: {memo['content_length']}, 作成日: {memo['created_at']}")

            # 空のコンテンツを持つメモを確認
            cur.execute("""
                SELECT COUNT(*) as count
                FROM memos
                WHERE is_deleted = FALSE
                  AND is_draft = FALSE
                  AND (content IS NULL OR TRIM(content) = '')
            """)
            empty_content = cur.fetchone()['count']
            print(f"\n空のコンテンツを持つメモ数: {empty_content}")

            # is_deleted, is_draftの値の分布を確認
            cur.execute("""
                SELECT is_deleted, is_draft, COUNT(*) as count
                FROM memos
                GROUP BY is_deleted, is_draft
                ORDER BY is_deleted, is_draft
            """)
            status_distribution = cur.fetchall()
            print(f"\nメモのステータス分布:")
            for row in status_distribution:
                print(f"  is_deleted: {row['is_deleted']}, is_draft: {row['is_draft']}, 件数: {row['count']}")

if __name__ == "__main__":
    main()