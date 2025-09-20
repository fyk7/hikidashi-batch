#!/usr/bin/env python3
"""
エンベディングプロバイダーのテストスクリプト
OpenAIとGeminiの両方をテストします
"""

import asyncio
import os
from dotenv import load_dotenv
from batch_create_embeddings import create_embedding

load_dotenv()

async def test_openai():
    """OpenAI エンベディングのテスト"""
    print("=== OpenAI エンベディングテスト ===")

    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY が設定されていません")
        return False

    try:
        embedding = await create_embedding(
            text="これはテストテキストです。",
            embedding_dimensions=512,
            provider="openai"
        )

        if embedding:
            print(f"✅ OpenAI成功: 次元数={len(embedding)}")
            print(f"   最初の5次元: {embedding[:5]}")
            return True
        else:
            print("❌ OpenAI失敗: エンベディング生成できませんでした")
            return False

    except Exception as e:
        print(f"❌ OpenAI エラー: {e}")
        return False

async def test_gemini():
    """Gemini エンベディングのテスト"""
    print("\n=== Gemini エンベディングテスト ===")

    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY が設定されていません")
        return False

    try:
        embedding = await create_embedding(
            text="これはテストテキストです。",
            embedding_dimensions=768,
            provider="gemini"
        )

        if embedding:
            print(f"✅ Gemini成功: 次元数={len(embedding)}")
            print(f"   最初の5次元: {embedding[:5]}")
            return True
        else:
            print("❌ Gemini失敗: エンベディング生成できませんでした")
            return False

    except Exception as e:
        print(f"❌ Gemini エラー: {e}")
        return False

async def main():
    """メインテスト関数"""
    print("エンベディングプロバイダーテストを開始します\n")

    openai_success = await test_openai()
    gemini_success = await test_gemini()

    print(f"\n=== テスト結果 ===")
    print(f"OpenAI: {'✅ 成功' if openai_success else '❌ 失敗'}")
    print(f"Gemini: {'✅ 成功' if gemini_success else '❌ 失敗'}")

    if openai_success or gemini_success:
        print("\n✅ 少なくとも1つのプロバイダーが正常に動作しています")
    else:
        print("\n❌ 全てのプロバイダーでエラーが発生しました")

if __name__ == "__main__":
    asyncio.run(main())