#!/usr/bin/env python3
"""
カテゴリーマイグレーションスクリプト
グローバルCategoryからユーザーごとのCategoryへ移行

Usage:
  python migrate_categories.py --status          # 現在の状態確認
  python migrate_categories.py --dry-run         # Dry Run
  python migrate_categories.py --execute         # 実行
"""

import os
import sys
from datetime import datetime
import psycopg2
import psycopg2.extras
import logging
from dotenv import load_dotenv
from utils import parse_db_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# データベース接続パラメータ
DB_PARAMS = parse_db_url(os.getenv('DATABASE_URL'))

def get_db_connection():
    """データベース接続を取得"""
    return psycopg2.connect(**DB_PARAMS)

def check_status(conn):
    logger.info("=" * 60)
    logger.info("マイグレーション状態")
    logger.info("=" * 60)

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cursor.execute("SELECT COUNT(*) as count FROM categories WHERE user_id IS NULL")
        old_cat = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM categories WHERE user_id IS NOT NULL")
        new_cat = cursor.fetchone()['count']

        cursor.execute("""
            SELECT COUNT(*) as count FROM memos m
            JOIN categories c ON c.id = m.category_id
            WHERE c.user_id IS NOT NULL
        """)
        new_memo = cursor.fetchone()['count']

        cursor.execute("""
            SELECT COUNT(*) as count FROM memos m
            JOIN categories c ON c.id = m.category_id
            WHERE c.user_id IS NULL
        """)
        old_memo = cursor.fetchone()['count']

        logger.info(f"旧カテゴリ: {old_cat}")
        logger.info(f"新カテゴリ: {new_cat}")
        logger.info(f"新カテゴリ参照Memo: {new_memo}")
        logger.info(f"旧カテゴリ参照Memo: {old_memo}")

        if new_cat == 0:
            logger.info("状態: 未実行")
        elif old_memo > 0:
            logger.info(f"状態: 移行中 ({old_memo}件のMemoが旧カテゴリ参照)")
        else:
            logger.info("状態: 移行完了")
    finally:
        cursor.close()

def migrate(conn, execute=False):
    mode = "実行" if execute else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"カテゴリー移行開始 ({mode})")
    logger.info("=" * 60)

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        # Step 1: 新カテゴリ作成
        logger.info("Step 1: 新カテゴリ作成中...")

        cursor.execute("""
            SELECT ucm.user_id, c.name, COUNT(*) as count
            FROM user_category_mapping ucm
            JOIN categories c ON c.id = ucm.category_id
            GROUP BY ucm.user_id, c.name
            ORDER BY ucm.user_id, c.name
        """)
        combinations = cursor.fetchall()
        logger.info(f"  対象: {len(combinations)}件")

        if execute:
            created = 0
            for combo in combinations:
                cursor.execute("""
                    INSERT INTO categories (id, user_id, name, depth, parent_id, created_at, updated_at)
                    VALUES (gen_random_uuid(), %s, %s, 0, NULL, NOW(), NOW())
                    ON CONFLICT (user_id, parent_id, name) DO NOTHING
                    RETURNING id
                """, (combo['user_id'], combo['name']))
                if cursor.fetchone():
                    created += 1
            logger.info(f"  作成: {created}件")
        else:
            logger.info("  (スキップ)")

        # Step 2: 対応表構築 & Memo更新
        logger.info("Step 2: Memo更新中...")

        cursor.execute("""
            SELECT
                old_cat.id AS old_id,
                new_cat.id AS new_id,
                ucm.user_id
            FROM user_category_mapping ucm
            JOIN categories old_cat ON old_cat.id = ucm.category_id
            JOIN categories new_cat ON new_cat.user_id = ucm.user_id
                AND new_cat.name = old_cat.name
                AND new_cat.parent_id IS NULL
            WHERE old_cat.user_id IS NULL
        """)
        mappings = cursor.fetchall()
        logger.info(f"  対応表: {len(mappings)}件")

        if len(mappings) == 0:
            logger.warning("  対応表が空です")
            return

        old_ids = [m['old_id'] for m in mappings]
        placeholders = ','.join(['%s'] * len(old_ids))
        cursor.execute(f"""
            SELECT COUNT(*) as count FROM memos
            WHERE category_id IN ({placeholders})
        """, old_ids)
        memo_count = cursor.fetchone()['count']
        logger.info(f"  更新対象: {memo_count}件のMemo")

        if execute:
            # バッチ更新（CASE式使用）
            BATCH_SIZE = 100
            updated = 0

            for i in range(0, len(mappings), BATCH_SIZE):
                batch = mappings[i:i+BATCH_SIZE]
                case_parts = []
                params = []

                for m in batch:
                    case_parts.append("WHEN category_id = %s AND user_id = %s THEN %s")
                    params.extend([m['old_id'], m['user_id'], m['new_id']])

                batch_old_ids = [m['old_id'] for m in batch]
                where_placeholders = ','.join(['%s'] * len(batch_old_ids))

                cursor.execute(f"""
                    UPDATE memos
                    SET category_id = CASE {" ".join(case_parts)} ELSE category_id END,
                        updated_at = NOW()
                    WHERE category_id IN ({where_placeholders})
                """, params + batch_old_ids)

                updated += cursor.rowcount

            logger.info(f"  更新完了: {updated}件")

            # 整合性チェック
            cursor.execute("""
                SELECT COUNT(*) as count FROM memos m
                LEFT JOIN categories c ON c.id = m.category_id
                WHERE m.category_id IS NOT NULL
                  AND (c.user_id IS NULL OR c.user_id != m.user_id)
            """)
            invalid = cursor.fetchone()['count']

            if invalid > 0:
                raise Exception(f"整合性エラー: {invalid}件のMemoが無効なカテゴリを参照")
            logger.info("  整合性チェック: OK")
        else:
            logger.info("  (スキップ)")
    finally:
        cursor.close()

    if execute:
        conn.commit()
        logger.info("✓ 移行完了")
        logger.info("")
        logger.info("次のステップ:")
        logger.info("  1. python verify_categories.py で検証")
        logger.info("  2. schema.prismaからUserCategoryモデル削除")
        logger.info("  3. npx prisma migrate dev")
    else:
        logger.info("")
        logger.info("実行する場合: python migrate_categories.py --execute")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate categories to per-user')
    parser.add_argument('--status', action='store_true', help='Show migration status')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')
    parser.add_argument('--execute', action='store_true', help='Execute migration')
    args = parser.parse_args()

    if not any([args.status, args.dry_run, args.execute]):
        parser.print_help()
        sys.exit(0)

    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)

        if args.status:
            check_status(conn)
        elif args.dry_run:
            migrate(conn, execute=False)
        elif args.execute:
            migrate(conn, execute=True)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    main()
