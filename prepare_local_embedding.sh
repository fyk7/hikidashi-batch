#!/bin/bash

# ローカル環境用のエンベディング準備スクリプト
# DATABASE_URLを固定して、チャンク作成とエンベディング生成を実行します

set -e  # エラーが発生したら即座に終了

# DATABASE_URLを固定
export DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:54322/postgres?schema=public"

# 1. メモをチャンクに分割してエンベディングを生成
uv run python memo_to_chunk.py --only-missing

# 2. メモとメモ画像のエンベディングを一括生成
python batch_create_embeddings.py --only-missing
