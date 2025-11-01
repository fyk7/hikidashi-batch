#!/usr/bin/env python3
"""
カテゴリーマイグレーション検証スクリプト

Usage:
  python verify_categories.py
"""

import os
import sys
import psycopg2
import psycopg2.extras
import logging
from dotenv import load_dotenv
from utils import parse_db_url

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# データベース接続パラメータ
DB_PARAMS = parse_db_url(os.getenv('DATABASE_URL'))

def get_db_connection():
    """データベース接続を取得"""
    return psycopg2.connect(**DB_PARAMS)

def verify(conn):
    logger.info("=" * 60)
    logger.info("カテゴリー移行検証")
    logger.info("=" * 60)

    results = []

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        # 検証1: Memoが正しいユーザーのカテゴリを参照
        cursor.execute("""
            SELECT COUNT(*) as count FROM memos m
            JOIN categories c ON c.id = m.category_id
            WHERE m.category_id IS NOT NULL
              AND c.user_id IS NOT NULL
              AND c.user_id != m.user_id
        """)
        mismatches = cursor.fetchone()['count']

        if mismatches > 0:
            logger.error(f"✗ Memoとカテゴリの所有者不一致: {mismatches}件")

            # 詳細情報を取得
            cursor.execute("""
                SELECT
                    m.id as memo_id,
                    m.user_id as memo_user_id,
                    m.title,
                    c.id as category_id,
                    c.user_id as category_user_id,
                    c.name as category_name
                FROM memos m
                JOIN categories c ON c.id = m.category_id
                WHERE m.category_id IS NOT NULL
                  AND c.user_id IS NOT NULL
                  AND c.user_id != m.user_id
                LIMIT 20
            """)
            invalid_memos = cursor.fetchall()
            logger.error("  無効なMemo一覧:")
            for memo in invalid_memos:
                logger.error(f"    Memo ID: {memo['memo_id']}, Title: {memo['title'][:50]}, "
                           f"Memo User: {memo['memo_user_id']}, Category: {memo['category_name']}, "
                           f"Category User: {memo['category_user_id']}")

            results.append(False)
        else:
            logger.info("✓ Memoとカテゴリの所有者一致")
            results.append(True)

        # 検証2: 旧カテゴリ参照がない
        cursor.execute("""
            SELECT COUNT(*) as count FROM memos m
            JOIN categories c ON c.id = m.category_id
            WHERE c.user_id IS NULL
        """)
        old_refs = cursor.fetchone()['count']

        if old_refs > 0:
            logger.error(f"✗ 旧カテゴリ参照: {old_refs}件")

            # 詳細情報を取得
            cursor.execute("""
                SELECT
                    m.id as memo_id,
                    m.user_id as memo_user_id,
                    m.title,
                    c.id as category_id,
                    c.name as category_name
                FROM memos m
                JOIN categories c ON c.id = m.category_id
                WHERE c.user_id IS NULL
                LIMIT 20
            """)
            old_ref_memos = cursor.fetchall()
            logger.error("  旧カテゴリを参照しているMemo一覧:")
            for memo in old_ref_memos:
                logger.error(f"    Memo ID: {memo['memo_id']}, Title: {memo['title'][:50]}, "
                           f"Memo User: {memo['memo_user_id']}, Old Category: {memo['category_name']} "
                           f"(ID: {memo['category_id']})")

            results.append(False)
        else:
            logger.info("✓ 旧カテゴリ参照なし")
            results.append(True)

        # 検証3: 孤立Memo
        cursor.execute("""
            SELECT COUNT(*) as count FROM memos m
            LEFT JOIN categories c ON c.id = m.category_id
            WHERE m.category_id IS NOT NULL AND c.id IS NULL
        """)
        orphans = cursor.fetchone()['count']

        if orphans > 0:
            logger.error(f"✗ 孤立Memo: {orphans}件")

            # 詳細情報を取得
            cursor.execute("""
                SELECT
                    m.id as memo_id,
                    m.user_id as memo_user_id,
                    m.title,
                    m.category_id
                FROM memos m
                LEFT JOIN categories c ON c.id = m.category_id
                WHERE m.category_id IS NOT NULL AND c.id IS NULL
                LIMIT 20
            """)
            orphan_memos = cursor.fetchall()
            logger.error("  孤立Memo一覧:")
            for memo in orphan_memos:
                logger.error(f"    Memo ID: {memo['memo_id']}, Title: {memo['title'][:50]}, "
                           f"Memo User: {memo['memo_user_id']}, "
                           f"Missing Category ID: {memo['category_id']}")

            results.append(False)
        else:
            logger.info("✓ 孤立Memoなし")
            results.append(True)

        # 検証4: カテゴリ重複
        cursor.execute("""
            SELECT COUNT(*) as count FROM (
                SELECT user_id, parent_id, name, COUNT(*) as cnt
                FROM categories
                WHERE user_id IS NOT NULL
                GROUP BY user_id, parent_id, name
                HAVING COUNT(*) > 1
            ) dups
        """)
        duplicates = cursor.fetchone()['count']

        if duplicates > 0:
            logger.error(f"✗ カテゴリ重複: {duplicates}件")

            # 詳細情報を取得
            cursor.execute("""
                SELECT
                    user_id,
                    parent_id,
                    name,
                    COUNT(*) as count,
                    array_agg(id) as category_ids
                FROM categories
                WHERE user_id IS NOT NULL
                GROUP BY user_id, parent_id, name
                HAVING COUNT(*) > 1
                LIMIT 20
            """)
            duplicate_categories = cursor.fetchall()
            logger.error("  重複カテゴリ一覧:")
            for cat in duplicate_categories:
                logger.error(f"    User: {cat['user_id']}, Name: {cat['name']}, "
                           f"Parent: {cat['parent_id']}, Count: {cat['count']}, "
                           f"IDs: {cat['category_ids']}")

            results.append(False)
        else:
            logger.info("✓ カテゴリ重複なし")
            results.append(True)

        # 検証5: user_category_mappingに存在しないカテゴリを参照しているメモ
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM memos m
            JOIN categories c ON c.id = m.category_id
            WHERE c.user_id IS NULL
              AND NOT EXISTS (
                SELECT 1
                FROM user_category_mapping ucm
                WHERE ucm.user_id = m.user_id
                  AND ucm.category_id = c.id
              )
        """)
        missing_mapping = cursor.fetchone()['count']

        if missing_mapping > 0:
            logger.error(f"✗ user_category_mappingに存在しないカテゴリを参照: {missing_mapping}件")

            # 詳細情報を取得
            cursor.execute("""
                SELECT
                    m.user_id,
                    c.id as category_id,
                    c.name as category_name,
                    COUNT(*) as memo_count
                FROM memos m
                JOIN categories c ON c.id = m.category_id
                WHERE c.user_id IS NULL
                  AND NOT EXISTS (
                    SELECT 1
                    FROM user_category_mapping ucm
                    WHERE ucm.user_id = m.user_id
                      AND ucm.category_id = c.id
                  )
                GROUP BY m.user_id, c.id, c.name
                ORDER BY memo_count DESC
                LIMIT 20
            """)
            missing_mappings = cursor.fetchall()
            logger.error("  user_category_mappingに存在しない組み合わせ:")
            for item in missing_mappings:
                logger.error(f"    User: {item['user_id']}, Category: {item['category_name']} "
                           f"(ID: {item['category_id']}), Memo数: {item['memo_count']}")

            # メモIDを取得して削除クエリを生成
            cursor.execute("""
                SELECT m.id
                FROM memos m
                JOIN categories c ON c.id = m.category_id
                WHERE c.user_id IS NULL
                  AND NOT EXISTS (
                    SELECT 1
                    FROM user_category_mapping ucm
                    WHERE ucm.user_id = m.user_id
                      AND ucm.category_id = c.id
                  )
            """)
            memo_ids = [row['id'] for row in cursor.fetchall()]

            logger.error("")
            logger.error("  削除対象のMemo ID一覧:")
            for memo_id in memo_ids:
                logger.error(f"    {memo_id}")

            logger.error("")
            logger.error("  削除クエリ:")
            if len(memo_ids) > 0:
                memo_ids_str = "', '".join(memo_ids)
                delete_query = f"DELETE FROM memos WHERE id IN ('{memo_ids_str}');"
                logger.error(f"  {delete_query}")

            results.append(False)
        else:
            logger.info("✓ 全てのメモがuser_category_mappingに登録されている")
            results.append(True)
    finally:
        cursor.close()

    logger.info("=" * 60)
    if all(results):
        logger.info("✓ 全検証合格")
        logger.info("")
        logger.info("次のステップ:")
        logger.info("  1. schema.prismaからUserCategoryモデル削除")
        logger.info("  2. npx prisma migrate dev --name remove_user_category")
        return 0
    else:
        logger.error("✗ 検証失敗")
        return 1

def main():
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        exit_code = verify(conn)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    main()
