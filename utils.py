import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_db_url(url):
    """データベースURLを解析して接続パラメータを取得する関数"""
    if not url:
        raise ValueError("データベースURLが設定されていません")
        
    parsed = urlparse(url)
    netloc = parsed.netloc
    
    if '@' not in netloc:
        raise ValueError("不正なデータベースURL形式です")
    
    # ユーザー名とパスワードを分離
    auth, host_part = netloc.split('@')
    user, password = auth.split(':')
    
    # ホストとポートを分離
    if ':' in host_part:
        host, port = host_part.split(':')
    else:
        host = host_part
        port = '5432'  # デフォルトのPostgreSQLポート
    
    # データベース名を取得
    dbname = parsed.path.strip('/')
    
    return {
        'dbname': dbname,
        'user': user,
        'password': password,
        'host': host,
        'port': port
    } 