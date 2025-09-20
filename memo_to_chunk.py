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
TARGET_CHUNK_SIZE = 700
MIN_CHUNK_SIZE = 500
OVERLAP_SIZE = 100
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

    def recursive_split(self, text: str, final_chunks: List[str]) -> None:
        """Recursively split long text by sentences"""
        if len(text) <= TARGET_CHUNK_SIZE:
            if text.strip():  # 空白のみでなければチャンクを生成
                final_chunks.append(text)
            return

        # Split by sentence delimiters
        sentence_pattern = r'(?<=[。\.！？\!?])\s+'
        sentences = re.split(sentence_pattern, text)

        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > TARGET_CHUNK_SIZE:
                if len(current_chunk) > MIN_CHUNK_SIZE:
                    final_chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence
            else:
                current_chunk += sentence

        if current_chunk.strip():  # 空白のみでなければチャンクを生成
            final_chunks.append(current_chunk)

    def split_html_into_chunks(self, html_content: str) -> List[str]:
        """Split HTML content into chunks with overlap"""
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove noise elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        # Primary split based on structure
        initial_chunks = []
        block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'pre', 'blockquote']

        for tag_name in block_tags:
            for element in soup.find_all(tag_name):
                text = element.get_text(strip=True)
                if text:
                    initial_chunks.append(text)

        # Secondary split based on character count
        sized_chunks = []
        for chunk in initial_chunks:
            self.recursive_split(chunk, sized_chunks)

        # HTMLからテキストが抽出できなかった場合、直接テキストを抽出して処理
        if not sized_chunks:
            text_content = soup.get_text(strip=True)
            if text_content.strip():  # 空白のみでなければチャンクを生成
                self.recursive_split(text_content, sized_chunks)

        # Add overlap
        if len(sized_chunks) <= 1:
            return sized_chunks

        final_chunks = [sized_chunks[0]]
        for i in range(1, len(sized_chunks)):
            prev_chunk = sized_chunks[i - 1]
            current_chunk = sized_chunks[i]

            # Add the last part of previous chunk to the beginning of current chunk
            overlap = prev_chunk[-OVERLAP_SIZE:] if len(prev_chunk) > OVERLAP_SIZE else prev_chunk
            final_chunks.append(overlap + "\n\n" + current_chunk)

        return final_chunks

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
                LEFT JOIN memo_chunks mc ON m.id = mc.memo_id AND mc.is_active = TRUE
                WHERE m.is_deleted = FALSE
                  AND m.is_draft = FALSE
                  AND mc.id IS NULL
                ORDER BY m.created_at DESC
                LIMIT 100
            """)
            return cur.fetchall()

    def deactivate_existing_chunks(self, conn, memo_id: str):
        """Deactivate existing chunks"""
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE memo_chunks
                SET is_active = FALSE, updated_at = NOW()
                WHERE memo_id = %s AND is_active = TRUE
            """, (memo_id,))

    def insert_chunks(self, conn, memo_id: str, chunks: List[str]) -> List[str]:
        """Insert chunks into database"""
        chunk_ids = []
        with conn.cursor() as cur:
            for i, chunk_text in enumerate(chunks):
                text_hash = self.generate_hash(chunk_text)

                # Check if chunk already exists
                cur.execute("""
                    SELECT id FROM memo_chunks
                    WHERE memo_id = %s AND text_hash = %s
                """, (memo_id, text_hash))

                existing = cur.fetchone()
                if existing:
                    # Reactivate existing chunk
                    cur.execute("""
                        UPDATE memo_chunks
                        SET is_active = TRUE, chunk_index = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (i, existing[0]))
                    chunk_ids.append(existing[0])
                else:
                    # Insert new chunk
                    cur.execute("""
                        INSERT INTO memo_chunks (id, memo_id, chunk_text, text_hash, chunk_index, is_active, created_at, updated_at)
                        VALUES (gen_random_uuid(), %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING id
                    """, (memo_id, chunk_text, text_hash, i))
                    chunk_ids.append(cur.fetchone()[0])

        return chunk_ids

    async def insert_embeddings(self, conn, chunk_ids: List[str], chunks: List[str], force_update: bool = True):
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

                # Generate embedding
                embedding = await self.generate_embedding(chunk_text)

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

        try:
            # Split HTML content into chunks
            chunks = self.split_html_into_chunks(content)

            if not chunks:
                logger.warning(f"No chunks generated for memo {memo_id}")
                return 0, 0

            # Deactivate existing chunks
            self.deactivate_existing_chunks(conn, memo_id)

            # Insert new chunks
            chunk_ids = self.insert_chunks(conn, memo_id, chunks)

            # Generate and save embeddings
            await self.insert_embeddings(conn, chunk_ids, chunks, force_update)

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
                              AND m.is_draft = FALSE
                            ORDER BY m.created_at DESC
                            LIMIT 100
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