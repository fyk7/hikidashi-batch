import asyncio
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional
import logging
import time
import os
import argparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from utils import parse_db_url
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAIのHTTPリクエストログを無効化
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# テーブル名定数
EMBEDDINGS_TABLE = 'embeddings'
IMAGE_EMBEDDINGS_TABLE = 'image_embeddings'

# 直接接続URLを使用（pgbouncerなしの方が適切）
DB_PARAMS = parse_db_url(os.getenv('DATABASE_URL'))


def truncate_text(text: str, max_chars: int = 2500) -> str:
    """テキストを指定文字数で切り詰める"""
    if len(text) <= max_chars:
        return text
    
    # 文字数制限を超える場合、適当な区切り位置で切り詰める
    truncated = text[:max_chars]
    
    # 最後の句読点や改行で切る
    for delimiter in ['\n\n', '\n', '。', '、', ' ']:
        last_delimiter = truncated.rfind(delimiter)
        if last_delimiter > max_chars * 0.8:  # 80%以上の位置にある区切り文字を使用
            return truncated[:last_delimiter + len(delimiter)]
    
    # 区切り文字が見つからない場合はそのまま切る
    return truncated

async def create_embedding(
    text: str,
    embedding_dimensions: int = 512,
    max_retries: int = 3,
    max_chars: int = 2500
) -> Optional[List[float]]:
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # テキストの長さ制限
    original_length = len(text)
    text = truncate_text(text, max_chars)
    if original_length > max_chars:
        logger.warning(f"テキストを切り詰めました: {original_length}文字 → {len(text)}文字")
    
    for retry_count in range(max_retries):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=embedding_dimensions
            )
            embedding = response.data[0].embedding
            
            # 次元数チェック
            if len(embedding) != embedding_dimensions:
                logger.warning(f"次元数不一致: 期待値={embedding_dimensions}, 実際={len(embedding)} (リトライ {retry_count + 1}/{max_retries})")
                if retry_count < max_retries - 1:
                    await asyncio.sleep(1)  # リトライ前に少し待機
                    continue
                else:
                    logger.error(f"次元数不一致が続いたため失敗: 期待値={embedding_dimensions}, 実際={len(embedding)}")
                    return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"エンベディング生成エラー (リトライ {retry_count + 1}/{max_retries}): {e}")
            if retry_count < max_retries - 1:
                await asyncio.sleep(1)  # リトライ前に少し待機
            else:
                logger.error(f"最大リトライ回数に達しました: {e}")
    
    return None

async def process_memo_batch(
    memo_batch: List[Dict[str, Any]],
    conn,
    force_update: bool = False,
    embedding_dimensions: int = 512,
    max_chars: int = 2500
) -> tuple[int, int]:
    cursor = conn.cursor()
    success_count = 0
    fail_count = 0
    
    for memo in memo_batch:
        memo_id = memo['id']
        title = memo.get('title', '')
        content = memo.get('content', '')
        
        combined_text = f"{title}\n{content}"
        embedding = await create_embedding(
            combined_text, 
            embedding_dimensions, 
            max_chars=max_chars
        )
        
        if not embedding:
            fail_count += 1
            continue
        
        try:
            if force_update:
                cursor.execute(
                    f"""INSERT INTO {EMBEDDINGS_TABLE} (
                        id, memo_id, vector, created_at, updated_at
                    ) VALUES (
                        gen_random_uuid(), %s, %s::vector, 
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (memo_id) 
                    DO UPDATE SET 
                        vector = EXCLUDED.vector, 
                        updated_at = CURRENT_TIMESTAMP""",
                    (memo_id, embedding)
                )
            else:
                cursor.execute(
                    f"SELECT 1 FROM {EMBEDDINGS_TABLE} WHERE memo_id = %s",
                    (memo_id,)
                )
                exists = cursor.fetchone() is not None
                
                if exists:
                    cursor.execute(
                        f"""UPDATE {EMBEDDINGS_TABLE} 
                        SET 
                            vector = %s::vector, 
                            updated_at = CURRENT_TIMESTAMP 
                        WHERE memo_id = %s""",
                        (embedding, memo_id)
                    )
                else:
                    cursor.execute(
                        f"""INSERT INTO {EMBEDDINGS_TABLE} (
                            id, memo_id, vector, created_at, updated_at
                        ) VALUES (
                            gen_random_uuid(), %s, %s::vector, 
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                        )""",
                        (memo_id, embedding)
                    )
            
            success_count += 1
        except Exception:
            fail_count += 1
    
    conn.commit()
    cursor.close()
    return success_count, fail_count

async def process_memo_image_batch(
    image_batch: List[Dict[str, Any]],
    conn,
    force_update: bool = False,
    embedding_dimensions: int = 512,
    max_chars: int = 2500
) -> tuple[int, int]:
    cursor = conn.cursor()
    success_count = 0
    fail_count = 0
    
    for image in image_batch:
        image_id = image['id']
        description = image.get('description', '')
        
        if not description or description.strip() == '':
            continue
        
        embedding = await create_embedding(
            description, 
            embedding_dimensions, 
            max_chars=max_chars
        )
        
        if not embedding:
            fail_count += 1
            continue
        
        try:
            if force_update:
                cursor.execute(
                    f"""INSERT INTO {IMAGE_EMBEDDINGS_TABLE} (
                        id, image_id, vector, created_at, updated_at
                    ) VALUES (
                        gen_random_uuid(), %s, %s::vector, 
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (image_id) 
                    DO UPDATE SET 
                        vector = EXCLUDED.vector, 
                        updated_at = CURRENT_TIMESTAMP""",
                    (image_id, embedding)
                )
            else:
                cursor.execute(
                    f"SELECT 1 FROM {IMAGE_EMBEDDINGS_TABLE} WHERE image_id = %s",
                    (image_id,)
                )
                exists = cursor.fetchone() is not None
                
                if exists:
                    cursor.execute(
                        f"""UPDATE {IMAGE_EMBEDDINGS_TABLE} 
                        SET 
                            vector = %s::vector, 
                            updated_at = CURRENT_TIMESTAMP 
                        WHERE image_id = %s""",
                        (embedding, image_id)
                    )
                else:
                    cursor.execute(
                        f"""INSERT INTO {IMAGE_EMBEDDINGS_TABLE} (
                            id, image_id, vector, created_at, updated_at
                        ) VALUES (
                            gen_random_uuid(), %s, %s::vector, 
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                        )""",
                        (image_id, embedding)
                    )
            
            success_count += 1
        except Exception:
            fail_count += 1
    
    conn.commit()
    cursor.close()
    return success_count, fail_count

async def validate_embedding_dimensions(
    conn,
    expected_dimensions: int
) -> None:
    """エンベディングの次元数が統一されているかチェックする"""
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # メモエンベディングの次元数チェック（pgvector用）
    try:
        cursor.execute(f"""
            SELECT COUNT(*) as total_count,
                   vector_dims(vector) as dimension
            FROM {EMBEDDINGS_TABLE}
            GROUP BY vector_dims(vector)
        """)
        memo_results = cursor.fetchall()
        
        if memo_results:
            logger.info("=== メモエンベディング次元数チェック ===")
            total_memo_embeddings = sum(row['total_count'] for row in memo_results)
            
            if len(memo_results) == 1 and memo_results[0]['dimension'] == expected_dimensions:
                logger.info(f"✅ メモエンベディング: 全{total_memo_embeddings}件が{expected_dimensions}次元で統一されています")
            else:
                logger.warning(f"⚠️  メモエンベディング次元数が不統一です:")
                for row in memo_results:
                    logger.warning(f"   {row['dimension']}次元: {row['total_count']}件")
                logger.warning(f"   期待値: {expected_dimensions}次元")
        else:
            logger.info("メモエンベディングが見つかりませんでした")
    except Exception as e:
        logger.error(f"メモエンベディング次元数チェックでエラー: {e}")
    
    # 画像エンベディングの次元数チェック（pgvector用）
    try:
        cursor.execute(f"""
            SELECT COUNT(*) as total_count,
                   vector_dims(vector) as dimension
            FROM {IMAGE_EMBEDDINGS_TABLE}
            GROUP BY vector_dims(vector)
        """)
        image_results = cursor.fetchall()
        
        if image_results:
            logger.info("=== 画像エンベディング次元数チェック ===")
            total_image_embeddings = sum(row['total_count'] for row in image_results)
            
            if len(image_results) == 1 and image_results[0]['dimension'] == expected_dimensions:
                logger.info(f"✅ 画像エンベディング: 全{total_image_embeddings}件が{expected_dimensions}次元で統一されています")
            else:
                logger.warning(f"⚠️  画像エンベディング次元数が不統一です:")
                for row in image_results:
                    logger.warning(f"   {row['dimension']}次元: {row['total_count']}件")
                logger.warning(f"   期待値: {expected_dimensions}次元")
        else:
            logger.info("画像エンベディングが見つかりませんでした")
    except Exception as e:
        logger.error(f"画像エンベディング次元数チェックでエラー: {e}")
    
    cursor.close()

async def fix_wrong_dimension_embeddings(
    target_dimensions: int,
    batch_size: int = 50,
    api_rate_limit_delay: float = 0.5,
    max_chars: int = 2500
) -> None:
    """間違った次元数のエンベディングを正しい次元数で修正する"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # メモエンベディングの次元数不一致を検出
        cursor.execute(f"""
            SELECT e.id, e.memo_id, m.title, m.content, vector_dims(e.vector) as current_dimension
            FROM {EMBEDDINGS_TABLE} e
            JOIN memos m ON e.memo_id = m.id
            WHERE vector_dims(e.vector) != %s
        """, (target_dimensions,))
        wrong_memo_embeddings = cursor.fetchall()
        
        # 画像エンベディングの次元数不一致を検出
        cursor.execute(f"""
            SELECT ie.id, ie.image_id, mi.description, vector_dims(ie.vector) as current_dimension
            FROM {IMAGE_EMBEDDINGS_TABLE} ie
            JOIN memo_images mi ON ie.image_id = mi.id
            WHERE vector_dims(ie.vector) != %s
        """, (target_dimensions,))
        wrong_image_embeddings = cursor.fetchall()
        
        if not wrong_memo_embeddings and not wrong_image_embeddings:
            logger.info(f"✅ 全てのエンベディングが{target_dimensions}次元で統一されています")
            return
        
        logger.info(f"次元数不一致の検出結果:")
        logger.info(f"  - メモエンベディング: {len(wrong_memo_embeddings)}件")
        logger.info(f"  - 画像エンベディング: {len(wrong_image_embeddings)}件")
        
        # メモエンベディングの修正
        if wrong_memo_embeddings:
            memo_success_count = 0
            memo_fail_count = 0
            
            memo_pbar = tqdm(total=len(wrong_memo_embeddings), desc="メモエンベディング修正", unit="件")
            
            for i, record in enumerate(wrong_memo_embeddings):
                embedding_id = record['id']
                memo_id = record['memo_id']
                title = record.get('title', '')
                content = record.get('content', '')
                current_dim = record['current_dimension']
                
                combined_text = f"{title}\n{content}"
                
                logger.info(f"修正中: メモID={memo_id}, 現在の次元={current_dim} → {target_dimensions}")
                
                new_embedding = await create_embedding(
                    combined_text, 
                    target_dimensions, 
                    max_chars=max_chars
                )
                
                if new_embedding:
                    try:
                        cursor.execute(
                            f"""UPDATE {EMBEDDINGS_TABLE} 
                            SET vector = %s::vector, updated_at = CURRENT_TIMESTAMP 
                            WHERE id = %s""",
                            (new_embedding, embedding_id)
                        )
                        conn.commit()
                        memo_success_count += 1
                        logger.info(f"✅ メモID={memo_id} の修正完了")
                    except Exception as e:
                        logger.error(f"❌ メモID={memo_id} の更新エラー: {e}")
                        memo_fail_count += 1
                else:
                    logger.error(f"❌ メモID={memo_id} の新エンベディング生成に失敗")
                    memo_fail_count += 1
                
                memo_pbar.update(1)
                
                if i + 1 < len(wrong_memo_embeddings):
                    await asyncio.sleep(api_rate_limit_delay)
            
            memo_pbar.close()
            logger.info(f"メモエンベディング修正完了: {memo_success_count}成功, {memo_fail_count}失敗")
        
        # 画像エンベディングの修正
        if wrong_image_embeddings:
            image_success_count = 0
            image_fail_count = 0
            
            image_pbar = tqdm(total=len(wrong_image_embeddings), desc="画像エンベディング修正", unit="件")
            
            for i, record in enumerate(wrong_image_embeddings):
                embedding_id = record['id']
                image_id = record['image_id']
                description = record.get('description', '')
                current_dim = record['current_dimension']
                
                if not description or description.strip() == '':
                    image_pbar.update(1)
                    continue
                
                logger.info(f"修正中: 画像ID={image_id}, 現在の次元={current_dim} → {target_dimensions}")
                
                new_embedding = await create_embedding(
                    description, 
                    target_dimensions, 
                    max_chars=max_chars
                )
                
                if new_embedding:
                    try:
                        cursor.execute(
                            f"""UPDATE {IMAGE_EMBEDDINGS_TABLE} 
                            SET vector = %s::vector, updated_at = CURRENT_TIMESTAMP 
                            WHERE id = %s""",
                            (new_embedding, embedding_id)
                        )
                        conn.commit()
                        image_success_count += 1
                        logger.info(f"✅ 画像ID={image_id} の修正完了")
                    except Exception as e:
                        logger.error(f"❌ 画像ID={image_id} の更新エラー: {e}")
                        image_fail_count += 1
                else:
                    logger.error(f"❌ 画像ID={image_id} の新エンベディング生成に失敗")
                    image_fail_count += 1
                
                image_pbar.update(1)
                
                if i + 1 < len(wrong_image_embeddings):
                    await asyncio.sleep(api_rate_limit_delay)
            
            image_pbar.close()
            logger.info(f"画像エンベディング修正完了: {image_success_count}成功, {image_fail_count}失敗")
        
        cursor.close()
        
        # 修正後の次元数チェック
        logger.info("=== 修正後チェック: エンベディング次元数の検証 ===")
        await validate_embedding_dimensions(conn, target_dimensions)
        
    except Exception as e:
        logger.error(f"次元数修正処理エラー: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

async def batch_create_embeddings(
    only_missing: bool = False,
    batch_size: int = 50,
    api_rate_limit_delay: float = 0.5,
    embedding_dimensions: int = 512,
    force_update: bool = False,
    max_chars: int = 2500
) -> None:
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Process memos
        if only_missing:
            # エンベディングがないメモのみを取得
            cursor.execute("""
                SELECT m.id, m.title, m.content 
                FROM memos m 
                LEFT JOIN embeddings e ON m.id = e.memo_id 
                WHERE e.memo_id IS NULL
                -- AND m.is_deleted = false
            """)
            all_memos = cursor.fetchall()
            logger.info(f"エンベディング未作成のメモ: {len(all_memos)}件")
        else:
            cursor.execute("""
                SELECT id, title, content
                FROM memos
                -- WHERE is_deleted = false
            """)
            all_memos = cursor.fetchall()
            logger.info(f"処理対象のメモ: {len(all_memos)}件")
            if force_update:
                logger.info("Force update mode: 既存のエンベディングを強制的に更新します")
        
        memo_success_total = 0
        memo_fail_total = 0
        
        # メモのエンベディング生成（進捗表示付き）
        if len(all_memos) > 0:
            memo_pbar = tqdm(total=len(all_memos), desc="メモエンベディング生成", unit="件", leave=True, dynamic_ncols=True)
            for i in range(0, len(all_memos), batch_size):
                batch = all_memos[i:i+batch_size]
                
                success, fail = await process_memo_batch(
                    [dict(memo) for memo in batch], 
                    conn, 
                    force_update, 
                    embedding_dimensions, 
                    max_chars
                )
                memo_success_total += success
                memo_fail_total += fail
                
                memo_pbar.update(len(batch))
                memo_pbar.refresh()  # 手動でリフレッシュ
                
                if i + batch_size < len(all_memos):
                    await asyncio.sleep(api_rate_limit_delay)
            memo_pbar.close()
        
        logger.info(f"メモエンベディング生成完了: {memo_success_total}/{len(all_memos)} 成功, {memo_fail_total} 失敗")

        # Process memo images with descriptions
        if only_missing:
            # エンベディングがない画像のみを取得
            cursor.execute(f"""
                SELECT mi.id, mi.description 
                FROM memo_images mi 
                LEFT JOIN {IMAGE_EMBEDDINGS_TABLE} ie ON mi.id = ie.image_id 
                WHERE ie.image_id IS NULL 
                AND mi.description IS NOT NULL 
                AND mi.description != ''
            """)
            all_images = cursor.fetchall()
            logger.info(f"エンベディング未作成の画像: {len(all_images)}件")
        else:
            cursor.execute("SELECT id, description FROM memo_images WHERE description IS NOT NULL AND description != ''")
            all_images = cursor.fetchall()
            logger.info(f"処理対象の画像: {len(all_images)}件")
        
        image_success_total = 0
        image_fail_total = 0
        
        # 画像のエンベディング生成（進捗表示付き）
        if len(all_images) > 0:
            image_pbar = tqdm(total=len(all_images), desc="画像エンベディング生成", unit="件")
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i+batch_size]
                
                success, fail = await process_memo_image_batch(
                    [dict(image) for image in batch], 
                    conn, 
                    force_update, 
                    embedding_dimensions, 
                    max_chars
                )
                image_success_total += success
                image_fail_total += fail
                
                image_pbar.update(len(batch))
                
                if i + batch_size < len(all_images):
                    await asyncio.sleep(api_rate_limit_delay)
            image_pbar.close()
        
        logger.info(f"画像エンベディング生成完了: {image_success_total}/{len(all_images)} 成功, {image_fail_total} 失敗")
        cursor.close()
        
        # 最終的な次元数チェック（全モードで実行）
        logger.info("=== 最終チェック: エンベディング次元数の検証 ===")
        await validate_embedding_dimensions(conn, embedding_dimensions)
        
    except Exception as e:
        logger.error(f"バッチ処理エラー: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='メモとメモ画像のエンベディングを一括生成・修正します')
    parser.add_argument('--only-missing', action='store_true', help='エンベディングが未作成のメモ・画像のみを処理します')
    parser.add_argument('--batch-size', type=int, default=50, help='バッチサイズ (デフォルト: 50)')
    parser.add_argument('--api-rate-limit-delay', type=float, default=0.5, help='API呼び出し間隔（秒）(デフォルト: 0.5)')
    parser.add_argument('--embedding-dimensions', type=int, default=512, help='エンベディングの次元数 (デフォルト: 512)')
    parser.add_argument('--force-update', action='store_true', help='既存のエンベディングを強制的に更新します')
    parser.add_argument('--fix-dimensions', action='store_true', help='間違った次元数のエンベディングを修正します')
    parser.add_argument('--max-chars', type=int, default=2500, help='エンベディング生成時のテキスト最大文字数 (デフォルト: 2500)')
    args = parser.parse_args()
    
    if args.fix_dimensions:
        logger.info(f"次元数修正モード: {args.embedding_dimensions}次元以外のエンベディングを修正します")
        asyncio.run(fix_wrong_dimension_embeddings(
            target_dimensions=args.embedding_dimensions,
            batch_size=args.batch_size,
            api_rate_limit_delay=args.api_rate_limit_delay,
            max_chars=args.max_chars
        ))
    else:
        if args.only_missing:
            logger.info("Missing embeddings only mode: エンベディング未作成のアイテムのみを処理します")
        if args.force_update:
            logger.info("Force update mode: 既存のエンベディングを強制的に更新します")
        
        asyncio.run(batch_create_embeddings(
            only_missing=args.only_missing,
            batch_size=args.batch_size,
            api_rate_limit_delay=args.api_rate_limit_delay,
            embedding_dimensions=args.embedding_dimensions,
            force_update=args.force_update,
            max_chars=args.max_chars
        ))

if __name__ == "__main__":
    main()
