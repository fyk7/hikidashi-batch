import psycopg2
import psycopg2.extras
import os
import logging
from dotenv import load_dotenv
from utils import parse_db_url
import uuid

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 元のデータベースと新しいデータベースの接続パラメータを取得
SOURCE_DB_PARAMS = parse_db_url(os.getenv('SOURCE_DATABASE_URL'))
TARGET_DB_PARAMS = parse_db_url(os.getenv('TARGET_DATABASE_URL'))

# 移行するテーブルのリスト（依存関係の順序に注意）
TABLES = [
    'categories',
    'user_category_mapping',
    'memos',
]

# 固定のユーザーID（環境変数から取得するか、ここで直接指定）
TARGET_USER_ID = os.getenv('TARGET_USER_ID', '00000000-0000-0000-0000-000000000000')

def migrate_table_with_batches(table, source_conn, target_conn, batch_size=100):
    """テーブルデータをバッチで移行する関数（固定ユーザーIDを使用）"""
    source_cursor = source_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    target_cursor = target_conn.cursor()
    
    # 総行数を取得
    source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total_rows = source_cursor.fetchone()[0]
    
    # テーブルのカラム情報を取得
    target_cursor.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table}'
    """)
    column_types = {row[0]: row[1] for row in target_cursor.fetchall()}
    
    offset = 0
    processed = 0
    skipped = 0
    
    while offset < total_rows:
        # バッチデータを取得
        source_cursor.execute(f"SELECT * FROM {table} LIMIT {batch_size} OFFSET {offset}")
        rows = source_cursor.fetchall()
        
        if not rows:
            break
            
        # カラム名を取得
        columns = [desc[0] for desc in source_cursor.description]
        
        # ユーザーIDを含むカラムを特定
        user_id_columns = [col for col in columns if col == 'user_id' or col.endswith('_user_id')]
        has_user_id = len(user_id_columns) > 0
        
        batch_success = True
        
        try:
            # 各行を処理
            for row in rows:
                # 行をディクショナリに変換
                row_dict = dict(zip(columns, row))
                
                # ユーザーIDを固定値に置き換え
                if has_user_id:
                    for col in user_id_columns:
                        if col in row_dict and row_dict[col] is not None:
                            row_dict[col] = TARGET_USER_ID
                
                # UUIDフィールドの処理
                for col in columns:
                    # UUIDフィールドの場合、新しいUUIDを生成
                    if col in column_types and column_types[col] == 'uuid' and row_dict[col] is not None:
                        # 主キーやユーザーID以外のUUIDフィールドは新しいUUIDに置き換え
                        if col != 'id' and col not in user_id_columns:
                            if not isinstance(row_dict[col], uuid.UUID) and not is_valid_uuid(row_dict[col]):
                                row_dict[col] = str(uuid.uuid4())
                
                # INSERT文を構築して実行
                placeholders = ', '.join(['%s'] * len(columns))
                column_names = ', '.join(columns)
                values = [row_dict[col] for col in columns]
                
                try:
                    target_cursor.execute(
                        f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})",
                        values
                    )
                except psycopg2.errors.UniqueViolation as e:
                    # 重複エラーの場合はスキップして続行
                    target_conn.rollback()  # このエラーのロールバック
                    logger.warning(f"{table}テーブルの重複エントリーをスキップ: {row_dict}")
                    skipped += 1
                    continue  # 次の行へ
                except Exception as e:
                    logger.error(f"{table}テーブルの行挿入エラー: {e}")
                    logger.error(f"問題のデータ: {row_dict}")
                    raise
            
            # バッチをコミット
            target_conn.commit()
            
            processed += len(rows) - skipped
            logger.info(f"{table}テーブル: {processed}/{total_rows}行を処理しました（{skipped}行スキップ）")
            skipped = 0  # バッチごとにリセット
            
        except Exception as e:
            # バッチ処理中にエラーが発生した場合
            target_conn.rollback()
            logger.error(f"{table}テーブルのバッチ処理エラー: {e}")
            batch_success = False
        
        # 次のバッチへ
        offset += batch_size
        
        # バッチが失敗した場合でも次のバッチを処理
        if not batch_success:
            logger.warning(f"{table}テーブル: オフセット {offset-batch_size} のバッチはスキップされました")
    
    source_cursor.close()
    target_cursor.close()
    logger.info(f"{table}テーブルの移行完了: 合計{processed}行処理、{skipped}行スキップ")

def is_valid_uuid(val):
    """文字列がUUID形式かどうかを確認する関数"""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError):
        return False

def migrate_data():
    """データを元のDBから新しいDBに移行する関数"""
    source_conn = None
    target_conn = None
    
    try:
        # 元のデータベースに接続
        logger.info("元のデータベースに接続中...")
        source_conn = psycopg2.connect(**SOURCE_DB_PARAMS)
        source_cursor = source_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 新しいデータベースに接続
        logger.info("新しいデータベースに接続中...")
        target_conn = psycopg2.connect(**TARGET_DB_PARAMS)
        target_cursor = target_conn.cursor()
        
        # グローバル定義されたTABLESを使用する
        for table in TABLES:
            logger.info(f"{table}テーブルのデータを移行中...")
            
            # user_category_mappingテーブルの場合は特別な処理を行う
            if table == 'user_category_mapping':
                migrate_user_category_mapping(source_conn, target_conn)
            else:
                migrate_table_with_batches(table, source_conn, target_conn)
        
        logger.info("すべてのデータの移行が完了しました")
    
    except Exception as e:
        logger.error(f"データ移行エラー: {e}")
        if target_conn:
            target_conn.rollback()
    
    finally:
        # 接続を閉じる
        if source_conn:
            source_cursor.close()
            source_conn.close()
        if target_conn:
            target_cursor.close()
            target_conn.close()

def migrate_user_category_mapping(source_conn, target_conn, batch_size=100):
    """user_category_mappingテーブルのデータを重複なしで移行する関数"""
    source_cursor = source_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    target_cursor = target_conn.cursor()
    
    # 重複のないcategory_idのみを取得
    source_cursor.execute("""
        SELECT DISTINCT category_id
        FROM user_category_mapping
    """)
    
    unique_categories = source_cursor.fetchall()
    total_rows = len(unique_categories)
    logger.info(f"user_category_mappingテーブル: {total_rows}件の一意のカテゴリーを検出")
    
    processed = 0
    skipped = 0
    
    # バッチ処理
    for i in range(0, total_rows, batch_size):
        batch = unique_categories[i:i+batch_size]
        
        try:
            for category_row in batch:
                category_id = category_row['category_id']
                
                # 新しいUUIDを生成
                new_id = str(uuid.uuid4())
                
                try:
                    target_cursor.execute(
                        """
                        INSERT INTO user_category_mapping (category_id, user_id, created_at, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (category_id, TARGET_USER_ID)
                    )
                except psycopg2.errors.UniqueViolation as e:
                    # 重複エラーの場合はスキップして続行
                    target_conn.rollback()  # このエラーのロールバック
                    logger.warning(f"user_category_mappingテーブルの重複エントリーをスキップ: category_id={category_id}")
                    skipped += 1
                    continue
                except Exception as e:
                    logger.error(f"user_category_mappingテーブルの行挿入エラー: {e}")
                    raise
            
            # バッチをコミット
            target_conn.commit()
            
            processed += len(batch) - skipped
            logger.info(f"user_category_mappingテーブル: {processed}/{total_rows}行を処理しました（{skipped}行スキップ）")
            skipped = 0  # バッチごとにリセット
            
        except Exception as e:
            target_conn.rollback()
            logger.error(f"user_category_mappingテーブルのバッチ処理エラー: {e}")
    
    source_cursor.close()
    target_cursor.close()
    logger.info(f"user_category_mappingテーブルの移行完了: 合計{processed}行処理、{skipped}行スキップ")

if __name__ == "__main__":
    migrate_data()
