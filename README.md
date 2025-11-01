# memo-chat-app-batch

## 概要
このプロジェクトは、メモチャットアプリケーションのバッチ処理を行うPythonプロジェクトです。

## 必要な環境
- uv（Pythonバージョン管理 + パッケージマネージャー）

## uvのインストール

### macOS/Linux（推奨）
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # パスを更新
uv --version  # インストール確認
```

### Homebrewを使用する場合
```sh
brew install uv
uv --version  # インストール確認
```

## プロジェクトのセットアップ

```sh
git clone <repository-url>
cd memo-chat-app-batch
uv python install 3.11  # Python 3.11をインストール
uv python pin 3.11  # プロジェクトで使用するバージョンを固定
uv sync  # 依存関係をインストール（.venv作成、uv.lock生成）
cp .env.example .env  # 環境変数ファイルを作成
# .envを編集して必要な環境変数を設定
```

## 基本的なコマンド

### スクリプトの実行
```sh
uv run python migrate_category_to_per_user.py  # スクリプトを実行
uv run python <script_name>.py  # 任意のスクリプトを実行
uv run python  # Python REPLを起動
```

### 依存関係の管理
```sh
uv add <package-name>  # パッケージを追加
uv add --dev <package-name>  # 開発用パッケージを追加
uv add "openai>=1.12.0"  # バージョン指定で追加
uv remove <package-name>  # パッケージを削除
uv sync --upgrade  # すべての依存関係を更新
uv add --upgrade <package-name>  # 特定のパッケージを更新
uv pip list  # インストール済みパッケージを確認
uv lock  # uv.lockファイルを再生成
```

### Pythonバージョン管理
```sh
uv python list  # インストール可能なPythonバージョンを確認
uv python install 3.11  # Python 3.11をインストール（どのディレクトリでもOK）
uv python install 3.12  # Python 3.12をインストール
uv python list --only-installed  # インストール済みのPythonを確認
uv python pin 3.11  # プロジェクトで使用するバージョンを固定（.python-version作成）
uv run python --version  # 現在のPythonバージョンを確認
```

**重要**: uvでインストールしたPythonは`~/.local/share/uv/python/`にグローバル保存され、どのディレクトリでインストールしても全プロジェクトで使用可能です。

### 仮想環境の操作
```sh
source .venv/bin/activate  # 仮想環境を有効化（macOS/Linux）
.venv\Scripts\activate  # 仮想環境を有効化（Windows）
rm -rf .venv && uv sync  # 仮想環境を削除して再作成
```

通常は`uv run`を使用するため、明示的な仮想環境の有効化は不要です。

## プロジェクトの依存関係

- `psycopg2`: PostgreSQLデータベースアダプタ
- `openai`: OpenAI APIクライアント
- `python-dotenv`: 環境変数管理
- `stripe`: Stripe決済API
- `google-generativeai`: Google Gemini APIクライアント
- `beautifulsoup4`: HTMLパーサー
- `numpy`: 数値計算ライブラリ
- `lxml`: XMLおよびHTMLパーサー

## uvの主な利点

1. **高速**: Rustで実装されており、Poetryよりも10-100倍高速
2. **シンプル**: Pythonバージョン管理とパッケージ管理を統合
3. **互換性**: `pyproject.toml`をそのまま使用可能
4. **オールインワン**: pyenv + poetry + pip-toolsの機能を1つのツールで実現

## Poetryからの移行

```sh
rm poetry.lock  # Poetryのロックファイルを削除（不要）
git add uv.lock .python-version  # uvのファイルをGit管理
```

**注意**: `pyproject.toml`のpoetry固有の設定は残しておいても問題ありません。uvは互換性があります。

## トラブルシューティング

### Pythonバージョンが正しく認識されない場合
```sh
cat .python-version  # .python-versionファイルを確認
uv run python --version  # 使用中のPythonバージョンを確認
uv python pin 3.11  # Pythonバージョンを再設定
```

### 依存関係の競合が発生した場合
```sh
rm uv.lock  # ロックファイルを削除
uv sync  # 再生成
```

### キャッシュのクリア
```sh
uv cache clean  # uvのキャッシュをクリア
```

### uvのPATH設定が正しくない場合
```sh
which uv  # どのuvが実行されているか確認
which -a uv  # すべてのuvの場所を確認
# ~/.zshrcまたは~/.bashrcに追加:
export PATH="$HOME/.local/bin:$PATH"
```

### pyenv環境からの移行
```sh
pyenv versions  # pyenvで管理していたPythonバージョンを確認
uv python install 3.11  # uvで同じバージョンをインストール
# ~/.zshrcや~/.bashrcからpyenvの設定をコメントアウト（必要に応じて）
```

## 参考リンク
- [uv公式ドキュメント](https://docs.astral.sh/uv/)
- [uvクイックスタート](https://docs.astral.sh/uv/getting-started/)
- [uvによるPythonバージョン管理](https://docs.astral.sh/uv/concepts/python-versions/)
- [pyproject.toml仕様](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
