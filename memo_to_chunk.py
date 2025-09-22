#!/usr/bin/env python3
"""
Memo chunk processor - Split memos into chunks and generate embeddings
"""

import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
import asyncio

from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.generativeai as genai
import numpy as np
from utils import parse_db_url

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_CHUNK_SIZE = 700  # 中心500文字 + 前後オーバーラップ分の余裕
OVERLAP_SIZE = 100  # 前後のオーバーラップ
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_GEMINI = "gemini-embedding-001"
BATCH_SIZE = 100

# Gemini API設定
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません。")
    genai.configure(api_key=api_key)


class MemoChunkProcessor:
    """Class for splitting memos into chunks and generating embeddings"""

    def __init__(self, provider: str = "gemini"):
        """Initialize processor"""
        self.provider = provider.lower()

        if self.provider == "openai":
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "gemini":
            try:
                configure_gemini()
            except ValueError as e:
                logger.error(f"Gemini API設定エラー: {e}")
                raise
        else:
            raise ValueError(f"サポートされていないプロバイダー: {provider}")

        self.db_config = parse_db_url(os.getenv('DATABASE_URL'))
        self._register_vector_type()

    def _register_vector_type(self):
        """Register PostgreSQL vector type"""
        def adapt_numpy_array(numpy_array):
            return AsIs(f"'{numpy_array.tolist()}'")
        register_adapter(np.ndarray, adapt_numpy_array)

    def connect_db(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def generate_hash(self, text: str) -> str:
        """Generate SHA-256 hash from text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def simple_chunk_split(self, text: str) -> List[str]:
        """テキストを固定サイズのチャンクに分割する（シンプル版）"""
        chunks = []
        position = 0

        logger.info(f"[SIMPLE_SPLIT] 入力テキスト長: {len(text)}文字")

        # 空文字またはトリムしても空の場合は空配列を返す
        if not text or text.strip() == '':
            logger.info("[SIMPLE_SPLIT] 空テキストのため分割なし")
            return chunks

        # 短いテキストの場合はそのまま1つのチャンクとして返す
        if len(text.strip()) <= TARGET_CHUNK_SIZE:
            logger.info(f"[SIMPLE_SPLIT] 短いテキスト（{len(text.strip())}文字）のため1チャンクで返却")
            chunks.append(text.strip())
            logger.info(f"[SIMPLE_SPLIT] 分割結果: {len(chunks)}個のチャンク")
            return chunks

        while position < len(text):
            # 残りが目標サイズ以下の場合は全て含める
            if len(text) - position <= TARGET_CHUNK_SIZE:
                chunk = text[position:]
                logger.info(f"[SIMPLE_SPLIT] 最終チャンク: {len(chunk)}文字")
                chunks.append(chunk)
                break

            # 目標サイズで切り出し
            end_position = position + TARGET_CHUNK_SIZE

            # 文の境界を探す（日本語・英語の句読点または改行）
            search_start = max(position + TARGET_CHUNK_SIZE - 50, position)
            search_end = min(position + TARGET_CHUNK_SIZE + 50, len(text))
            search_text = text[search_start:search_end]

            # 句読点または改行を探す（日本語と英語の両方に対応）
            boundary_chars = ['。', '！', '？', '.', '!', '?', '\n']
            last_boundary_pos = -1

            for char in boundary_chars:
                pos = search_text.rfind(char)
                if pos > last_boundary_pos:
                    last_boundary_pos = pos

            if last_boundary_pos != -1:
                end_position = search_start + last_boundary_pos + 1

            chunk = text[position:end_position]
            logger.info(f"[SIMPLE_SPLIT] チャンク[{len(chunks)}]: {len(chunk)}文字")
            chunks.append(chunk)
            position = end_position

        logger.info(f"[SIMPLE_SPLIT] 分割結果: {len(chunks)}個のチャンク")
        return chunks

    def strip_html_tags(self, html_content: str) -> str:
        """HTMLタグを除去してプレーンテキストを取得"""
        soup = BeautifulSoup(html_content, 'lxml')
        return soup.get_text(strip=True)

    def split_html_into_chunks(self, html_content: str) -> List[str]:
        """HTMLコンテンツをチャンク分割し、オーバーラップを追加する"""
        newline_char = '\\n'  # Define at function start for use in f-strings
        logger.info(f"[CHUNK_SPLIT] HTMLコンテンツ長: {len(html_content)}文字")
        logger.info(f"[CHUNK_SPLIT] HTMLコンテンツ内容: \"{html_content[:200].replace(chr(10), newline_char)}...\"")

        soup = BeautifulSoup(html_content, 'lxml')

        # 1. ノイズとなる要素を削除
        for tag in soup.find_all(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        # 2. 構造に基づいた一次分割
        initial_chunks = []
        block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'pre', 'blockquote']
        seen_texts = set()  # 重複チェック用

        for tag_name in block_tags:
            for element in soup.find_all(tag_name):
                text = element.get_text(strip=True)
                if text and text not in seen_texts:  # 重複チェック追加
                    logger.info(f"[CHUNK_SPLIT] 初期ブロック抽出[{len(initial_chunks)}]: \"{text[:100].replace(chr(10), newline_char)}...\" ({len(text)}文字)")
                    initial_chunks.append(text)
                    seen_texts.add(text)
                elif text and text in seen_texts:
                    logger.warning(f"[CHUNK_SPLIT] 重複ブロックをスキップ: \"{text[:50].replace(chr(10), newline_char)}...\"")

        logger.info(f"[CHUNK_SPLIT] 初期ブロック数: {len(initial_chunks)}")

        # 3. 全ブロックを結合してから分割
        full_text = '\n\n'.join(initial_chunks)

        # ブロックが見つからない場合、HTMLタグを除去してプレーンテキストとして処理
        if full_text.strip() == '' and html_content.strip() != '':
            logger.info("[CHUNK_SPLIT] HTMLブロックが見つからないため、プレーンテキストとして処理")
            full_text = self.strip_html_tags(html_content)
            logger.info(f"[CHUNK_SPLIT] プレーンテキスト化後: {len(full_text)}文字")

        logger.info(f"[CHUNK_SPLIT] 結合後の総文字数: {len(full_text)}文字")

        # シンプルな固定サイズ分割を実行
        sized_chunks = self.simple_chunk_split(full_text)

        logger.info(f"[CHUNK_SPLIT] サイズ調整後チャンク数: {len(sized_chunks)}")

        # 4. オーバーラップの追加
        if len(sized_chunks) <= 1:
            logger.info(f"[CHUNK_SPLIT] オーバーラップなし(チャンク数 <= 1): {len(sized_chunks)}")
            return sized_chunks

        final_chunks_with_overlap = [sized_chunks[0]]
        logger.info(f"[CHUNK_SPLIT] オーバーラップ追加開始: {OVERLAP_SIZE}文字")

        for i in range(1, len(sized_chunks)):
            prev_chunk = sized_chunks[i - 1]
            current_chunk = sized_chunks[i]

            # 前のチャンクの末尾を取得し、空白や改行でトリム
            overlap = prev_chunk[-OVERLAP_SIZE:].strip()

            # 現在のチャンクが既に同じ内容で始まっているかチェック
            current_start = current_chunk[:min(len(overlap) + 20, len(current_chunk))].strip()
            if overlap in current_start or current_start[:30] in overlap:
                logger.info(f"[CHUNK_SPLIT] チャンク[{i}]: オーバーラップ重複検出、スキップ")
                final_chunks_with_overlap.append(current_chunk)
            else:
                chunk_with_overlap = overlap + "\n\n" + current_chunk
                logger.info(f"[CHUNK_SPLIT] チャンク[{i}]: オーバーラップ長={len(overlap)}, 結果長={len(chunk_with_overlap)}")
                logger.info(f"[CHUNK_SPLIT] チャンク[{i}]内容: \"{chunk_with_overlap[:150].replace(chr(10), newline_char)}...\"")
                final_chunks_with_overlap.append(chunk_with_overlap)

        logger.info(f"[CHUNK_SPLIT] 最終チャンク数: {len(final_chunks_with_overlap)}")

        # 重複チェック
        hash_set = set()
        duplicates = []
        for index, chunk in enumerate(final_chunks_with_overlap):
            chunk_hash = self.generate_hash(chunk)
            if chunk_hash in hash_set:
                duplicates.append({'index': index, 'hash': chunk_hash[:8]})
                logger.warning(f"[CHUNK_SPLIT] 重複チャンク検出: インデックス={index}, ハッシュ={chunk_hash[:8]}...")
                logger.warning(f"[CHUNK_SPLIT] 重複内容: \"{chunk[:200].replace(chr(10), newline_char)}...\"")
            else:
                hash_set.add(chunk_hash)

        if duplicates:
            logger.warning(f"[CHUNK_SPLIT] 重複チャンク {len(duplicates)}個検出")
        else:
            logger.info("[CHUNK_SPLIT] 重複チャンクなし")

        return final_chunks_with_overlap

    async def generate_embedding_openai(self, text: str) -> List[float]:
        """Generate embedding from text using OpenAI"""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL_OPENAI
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation error: {e}")
            raise

    def generate_embedding_gemini(self, text: str) -> List[float]:
        """Generate embedding from text using Gemini"""
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL_GEMINI,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            )
            embedding = result['embedding']

            # 768次元であることを確認
            if len(embedding) != 768:
                raise ValueError(f"Expected 768 dimensions, but got {len(embedding)} dimensions from Gemini embedding")

            logger.debug(f"Generated Gemini embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Gemini embedding generation error: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding from text"""
        if self.provider == "openai":
            return await self.generate_embedding_openai(text)
        elif self.provider == "gemini":
            # Geminiは同期関数なので、asyncioで実行
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.generate_embedding_gemini,
                text
            )
        else:
            raise ValueError(f"サポートされていないプロバイダー: {self.provider}")

    def get_unprocessed_memos(self, conn) -> List[Dict]:
        """Get memos that haven't been processed into chunks"""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT m.id, m.title, m.content
                FROM memos m
                LEFT JOIN memo_chunks mc ON m.id = mc.memo_id
                WHERE m.is_deleted = FALSE
                  -- AND m.is_draft = FALSE
                  AND mc.id IS NULL
                ORDER BY m.created_at DESC
            """)
            return cur.fetchall()

    def delete_existing_chunks(self, conn, memo_id: str):
        """Delete existing chunks for a memo"""
        with conn.cursor() as cur:
            # First delete embeddings for the chunks
            cur.execute("""
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (
                    SELECT id FROM memo_chunks WHERE memo_id = %s
                )
            """, (memo_id,))

            # Then delete the chunks themselves
            cur.execute("""
                DELETE FROM memo_chunks
                WHERE memo_id = %s
            """, (memo_id,))

    def insert_chunks(self, conn, memo_id: str, chunks: List[str]) -> List[str]:
        """Insert chunks into database"""
        chunk_ids = []
        with conn.cursor() as cur:
            for i, chunk_text in enumerate(chunks):
                text_hash = self.generate_hash(chunk_text)

                # Insert new chunk
                cur.execute("""
                    INSERT INTO memo_chunks (id, memo_id, chunk_text, text_hash, chunk_index, created_at, updated_at)
                    VALUES (gen_random_uuid(), %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (memo_id, chunk_text, text_hash, i))
                chunk_ids.append(cur.fetchone()[0])

        return chunk_ids

    async def insert_embeddings(self, conn, chunk_ids: List[str], chunks: List[str], memo_title: str = "", force_update: bool = True):
        """Generate and save chunk embeddings"""
        embeddings_data = []

        for chunk_id, chunk_text in zip(chunk_ids, chunks):
            try:
                # Check if embedding already exists
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM chunk_embeddings
                        WHERE chunk_id = %s
                    """, (chunk_id,))

                    existing = cur.fetchone()
                    if existing and not force_update:
                        continue  # Skip if embedding already exists and not forcing update

                # メモのタイトルを先頭に付与してコンテキストを追加
                text_with_context = f"{memo_title}\n\n{chunk_text}" if memo_title else chunk_text
                logger.info(f"[EMBEDDING] チャンクID: {chunk_id} - テキスト長: {len(text_with_context)}文字 (タイトル{'含む' if memo_title else '含まず'})")

                # Generate embedding
                embedding = await self.generate_embedding(text_with_context)

                # 追加の次元数チェック（保険）
                if len(embedding) != 768:
                    logger.error(f"Invalid embedding dimensions for chunk {chunk_id}: {len(embedding)} (expected 768)")
                    continue

                embeddings_data.append((chunk_id, embedding))

            except Exception as e:
                logger.error(f"Embedding generation error for chunk {chunk_id}: {e}")
                continue

        # Insert or update embeddings in batch
        if embeddings_data:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """
                    INSERT INTO chunk_embeddings (id, chunk_id, vector, created_at)
                    VALUES (gen_random_uuid(), %s, %s::vector, CURRENT_TIMESTAMP)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        vector = EXCLUDED.vector
                    """,
                    embeddings_data,
                    page_size=BATCH_SIZE
                )

            logger.info(f"Successfully inserted/updated {len(embeddings_data)} embeddings, all with 768 dimensions")

    def validate_embedding_dimensions(self, conn) -> None:
        """Validate that all embeddings in the database are 768 dimensions"""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT COUNT(*) as total_count,
                           vector_dims(vector) as dimension
                    FROM chunk_embeddings
                    GROUP BY vector_dims(vector)
                """)
                results = cur.fetchall()

                if results:
                    logger.info("=== チャンクエンベディング次元数チェック ===")
                    total_embeddings = sum(row['total_count'] for row in results)

                    if len(results) == 1 and results[0]['dimension'] == 768:
                        logger.info(f"✅ 全{total_embeddings}件のチャンクエンベディングが768次元で統一されています")
                    else:
                        logger.warning(f"⚠️  チャンクエンベディング次元数が不統一です:")
                        for row in results:
                            logger.warning(f"   {row['dimension']}次元: {row['total_count']}件")
                        logger.warning(f"   期待値: 768次元")
                else:
                    logger.info("チャンクエンベディングが見つかりませんでした")
            except Exception as e:
                logger.error(f"チャンクエンベディング次元数チェックでエラー: {e}")

    async def process_memo(self, conn, memo: Dict, force_update: bool = True) -> Tuple[int, int]:
        """Process a single memo"""
        memo_id = memo['id']
        content = memo['content']
        title = memo.get('title', '')  # memoのtitleを取得

        try:
            # Split HTML content into chunks
            chunks = self.split_html_into_chunks(content)

            if not chunks:
                logger.warning(f"No chunks generated for memo {memo_id}")
                return 0, 0

            # Delete existing chunks
            self.delete_existing_chunks(conn, memo_id)

            # Insert new chunks
            chunk_ids = self.insert_chunks(conn, memo_id, chunks)

            # Generate and save embeddings with title context
            await self.insert_embeddings(conn, chunk_ids, chunks, title, force_update)

            conn.commit()

            logger.info(f"Processed memo {memo_id}: {len(chunks)} chunks created")
            return len(chunks), len(chunk_ids)

        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing memo {memo_id}: {e}")
            raise

    async def run(self, force_update: bool = True):
        """Execute main processing"""
        total_memos = 0
        total_chunks = 0
        total_embeddings = 0

        try:
            with self.connect_db() as conn:
                # Get unprocessed memos (or all memos if force_update)
                if force_update:
                    # Get all memos if force updating
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            SELECT m.id, m.title, m.content
                            FROM memos m
                            WHERE m.is_deleted = FALSE
                              -- AND m.is_draft = FALSE
                            ORDER BY m.created_at DESC
                        """)
                        memos = cur.fetchall()
                else:
                    memos = self.get_unprocessed_memos(conn)

                if not memos:
                    logger.info("No memos to process")
                    return

                logger.info(f"Starting to process {len(memos)} memos (force_update={force_update})")

                for memo in memos:
                    try:
                        chunks_created, embeddings_created = await self.process_memo(conn, memo, force_update)
                        total_memos += 1
                        total_chunks += chunks_created
                        total_embeddings += embeddings_created

                    except Exception as e:
                        logger.error(f"Failed to process memo {memo['id']}: {e}")
                        continue

                logger.info(
                    f"Processing complete: {total_memos} memos, "
                    f"{total_chunks} chunks, "
                    f"{total_embeddings} embeddings"
                )

                # 最終的な次元数チェック
                logger.info("=== 最終チェック: チャンクエンベディング次元数の検証 ===")
                self.validate_embedding_dimensions(conn)

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='メモをチャンクに分割してエンベディングを生成します')
    parser.add_argument('--provider', type=str, default='gemini', choices=['openai', 'gemini'],
                       help='エンベディングプロバイダー (デフォルト: gemini)')
    parser.add_argument('--force-update', action='store_true',
                       help='既存のチャンクとエンベディングを強制的に更新します')
    parser.add_argument('--only-missing', action='store_true',
                       help='チャンクが未作成のメモのみを処理します（force-updateと排他）')
    args = parser.parse_args()

    # force-updateとonly-missingは排他的
    if args.force_update and args.only_missing:
        logger.error("--force-update と --only-missing は同時に指定できません")
        return

    force_update = args.force_update or not args.only_missing  # デフォルトはforce_update=True

    logger.info(f"エンベディングプロバイダー: {args.provider}")
    logger.info(f"強制更新: {force_update}")

    processor = MemoChunkProcessor(provider=args.provider)
    await processor.run(force_update=force_update)


if __name__ == "__main__":
    asyncio.run(main())