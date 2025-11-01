#!/usr/bin/env python3
"""
旧カテゴリ（共有カテゴリ）を削除するクリーンアップスクリプト

migrate_categories.py 実行後に残った旧カテゴリ（user_id が NULL）を削除します。
新カテゴリはすでに各ユーザー用に作成済みで、メモも移行済みのため、
旧カテゴリは不要になっています。

Usage:
  python update_category_user_ids.py --status          # 現在の状態確認
  python update_category_user_ids.py --dry-run         # Dry Run
  python update_category_user_ids.py --execute         # 実行
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
        logging.FileHandler(f'update_category_user_ids_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    logger.info("旧カテゴリ（共有カテゴリ）削除状態")
    logger.info("=" * 60)

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        # NULL の category 数（旧カテゴリ）
        cursor.execute("SELECT COUNT(*) as count FROM categories WHERE user_id IS NULL")
        old_categories_count = cursor.fetchone()['count']

        # NOT NULL の category 数（新カテゴリ）
        cursor.execute("SELECT COUNT(*) as count FROM categories WHERE user_id IS NOT NULL")
        new_categories_count = cursor.fetchone()['count']

        # user_category_mapping の総数
        cursor.execute("SELECT COUNT(*) as count FROM user_category_mapping")
        mapping_count = cursor.fetchone()['count']

        # 旧カテゴリの詳細
        if old_categories_count > 0:
            cursor.execute("""
                SELECT c.id, c.name,
                       CASE WHEN ucm.category_id IS NOT NULL THEN true ELSE false END as has_mapping
                FROM categories c
                LEFT JOIN user_category_mapping ucm ON ucm.category_id = c.id
                WHERE c.user_id IS NULL
                GROUP BY c.id, c.name, ucm.category_id
                ORDER BY c.name
            """)
            old_categories = cursor.fetchall()

            with_mapping = sum(1 for c in old_categories if c['has_mapping'])
            without_mapping = old_categories_count - with_mapping

        logger.info(f"新カテゴリ（user_id設定済み）: {new_categories_count}件")
        logger.info(f"旧カテゴリ（user_id = NULL）: {old_categories_count}件")
        logger.info(f"  - user_category_mappingに存在: {with_mapping if old_categories_count > 0 else 0}件")
        logger.info(f"  - 孤立カテゴリ: {without_mapping if old_categories_count > 0 else 0}件")

        if old_categories_count == 0:
            logger.info("✓ 状態: 旧カテゴリはすべて削除済み")
        else:
            logger.warning(f"⚠ 状態: {old_categories_count}件の旧カテゴリが残っています（削除推奨）")
            logger.info("")
            logger.info("旧カテゴリ一覧:")
            for cat in old_categories[:15]:
                mapping_status = "共有カテゴリ" if cat['has_mapping'] else "孤立"
                logger.info(f"  - '{cat['name']}' (ID: {cat['id']}) [{mapping_status}]")
            if old_categories_count > 15:
                logger.info(f"  ... 他 {old_categories_count - 15}件")

    finally:
        cursor.close()

def delete_old_categories(conn, execute=False):
    mode = "実行" if execute else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"旧カテゴリ削除開始 ({mode})")
    logger.info("=" * 60)

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        # 削除対象の旧カテゴリを取得
        cursor.execute("""
            SELECT c.id, c.name,
                   CASE WHEN ucm.category_id IS NOT NULL THEN true ELSE false END as has_mapping
            FROM categories c
            LEFT JOIN user_category_mapping ucm ON ucm.category_id = c.id
            WHERE c.user_id IS NULL
            GROUP BY c.id, c.name, ucm.category_id
            ORDER BY c.name
        """)
        old_categories = cursor.fetchall()

        if len(old_categories) == 0:
            logger.info("削除対象の旧カテゴリがありません")
            return

        logger.info(f"削除対象: {len(old_categories)}件の旧カテゴリ")
        logger.info("")

        # カテゴリ一覧を表示
        for cat in old_categories[:15]:
            mapping_status = "共有カテゴリ" if cat['has_mapping'] else "孤立"
            logger.info(f"  - '{cat['name']}' (ID: {cat['id']}) [{mapping_status}]")
        if len(old_categories) > 15:
            logger.info(f"  ... 他 {len(old_categories) - 15}件")

        logger.info("")

        if execute:
            # すべての旧カテゴリを削除
            old_category_ids = [cat['id'] for cat in old_categories]
            placeholders = ','.join(['%s'] * len(old_category_ids))
            cursor.execute(f"""
                DELETE FROM categories
                WHERE id IN ({placeholders})
            """, old_category_ids)
            deleted_count = cursor.rowcount
            logger.info(f"✓ 削除完了: {deleted_count}件のカテゴリを削除しました")

            # 削除後の確認
            cursor.execute("SELECT COUNT(*) as count FROM categories WHERE user_id IS NULL")
            remaining = cursor.fetchone()['count']

            if remaining == 0:
                logger.info("✓ すべての旧カテゴリが削除されました！")
            else:
                logger.warning(f"⚠ まだ{remaining}件の旧カテゴリが残っています")
        else:
            logger.info("(DRY RUN: 実際の削除はスキップ)")
            logger.info(f"実行時に {len(old_categories)}件のカテゴリが削除されます")

    finally:
        cursor.close()

    if execute:
        conn.commit()
        logger.info("=" * 60)
        logger.info("✓ 処理完了")
        logger.info("")
        logger.info("次のステップ:")
        logger.info("  1. python update_category_user_ids.py --status で確認")
        logger.info("  2. Webアプリのディレクトリで npx prisma migrate dev --name finalize_category_migration")
        logger.info("     → UserCategoryテーブルを削除（user_idはすでにNOT NULL制約済み）")
    else:
        logger.info("")
        logger.info("実行する場合: python update_category_user_ids.py --execute")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Delete old shared categories (user_id = NULL)')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')
    parser.add_argument('--execute', action='store_true', help='Execute deletion')
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
            delete_old_categories(conn, execute=False)
        elif args.execute:
            delete_old_categories(conn, execute=True)

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
