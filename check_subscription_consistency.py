import os
import stripe
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from enum import Enum

# .envファイルから環境変数をロード
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stripeの初期化
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
if not stripe.api_key:
    raise ValueError("STRIPE_SECRET_KEY環境変数が設定されていません")

# データベース接続情報
def parse_db_url(url):
    """DATABASE_URLを解析して接続パラメータを返す"""
    if not url:
        raise ValueError("DATABASE_URL環境変数が設定されていません")
    
    # postgresqlの形式: postgresql://username:password@hostname:port/database
    parts = url.split("://")[1].split("@")
    user_pass = parts[0].split(":")
    host_port_db = parts[1].split("/")
    host_port = host_port_db[0].split(":")
    
    return {
        "user": user_pass[0],
        "password": user_pass[1] if len(user_pass) > 1 else "",
        "host": host_port[0],
        "port": int(host_port[1]) if len(host_port) > 1 else 5432,
        "database": host_port_db[1]
    }

DB_PARAMS = parse_db_url(os.getenv('DATABASE_URL'))

# プランタイプとサブスクリプションステータスのEnum
class PlanType(Enum):
    FREE = "FREE"
    BASIC = "BASIC"
    PRO = "PRO"

class SubscriptionStatus(Enum):
    ACTIVE = "ACTIVE"
    INCOMPLETE = "INCOMPLETE"
    PAST_DUE = "PAST_DUE"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"

# プランタイプに基づいて月間トークン上限を取得
def get_monthly_token_limit(plan_type: PlanType) -> int:
    if plan_type == PlanType.PRO:
        return 5000000  # 500万トークン
    elif plan_type == PlanType.BASIC:
        return 1000000  # 100万トークン
    else:
        return 50000    # 5万トークン (FREE)

# Stripeの価格IDからプランタイプを判定
def determine_plan_type(price_id: str) -> PlanType:
    pro_plan_price_id = os.getenv("STRIPE_PRO_PLAN_PRICE_ID")
    basic_plan_price_id = os.getenv("STRIPE_BASIC_PLAN_PRICE_ID")
    
    if price_id == pro_plan_price_id:
        return PlanType.PRO
    elif price_id == basic_plan_price_id:
        return PlanType.BASIC
    elif price_id and "PRO" in price_id.upper():
        return PlanType.PRO
    else:
        return PlanType.BASIC

# Stripeのステータスをアプリのステータスに変換
def determine_subscription_status(stripe_status: str, cancel_at_period_end: bool) -> SubscriptionStatus:
    if stripe_status in ["canceled", "unpaid", "incomplete_expired"]:
        return SubscriptionStatus.CANCELED
    elif cancel_at_period_end and stripe_status != "canceled":
        return SubscriptionStatus.CANCELED
    else:
        return SubscriptionStatus.ACTIVE

class StripeSubscriptionProvider:
    """Stripe APIからサブスクリプション情報を取得するクラス"""
    
    @staticmethod
    def fetch_all_subscriptions() -> List[Dict[str, Any]]:
        """Stripeからすべてのサブスクリプションを取得"""
        logger.info("Stripeからすべてのサブスクリプションを取得しています...")
        subscriptions = []
        has_more = True
        starting_after = None
        
        while has_more:
            params = {"limit": 100}
            if starting_after:
                params["starting_after"] = starting_after
                
            response = stripe.Subscription.list(**params)
            subscriptions.extend(response["data"])
            has_more = response["has_more"]
            
            if has_more and response["data"]:
                starting_after = response["data"][-1]["id"]
            else:
                has_more = False
        
        logger.info(f"{len(subscriptions)}件のサブスクリプションが見つかりました")
        return subscriptions
    
    @staticmethod
    def format_subscription_data(stripe_sub: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Stripeのサブスクリプションデータをフォーマット"""
        price_id = stripe_sub["items"]["data"][0]["price"]["id"] if stripe_sub["items"]["data"] else None
        return {
            "stripe_subscription_id": stripe_sub["id"],
            "customer_id": stripe_sub["customer"],
            "user_id": user_id,
            "status": stripe_sub["status"],
            "price_id": price_id,
            "current_period_start": datetime.fromtimestamp(stripe_sub["current_period_start"]),
            "current_period_end": datetime.fromtimestamp(stripe_sub["current_period_end"]),
            "cancel_at_period_end": stripe_sub["cancel_at_period_end"],
            "canceled_at": datetime.fromtimestamp(stripe_sub["canceled_at"]) if stripe_sub.get("canceled_at") else None,
            "plan_type": determine_plan_type(price_id).value,
            "app_status": determine_subscription_status(stripe_sub["status"], stripe_sub["cancel_at_period_end"]).value,
            "monthly_token_limit": get_monthly_token_limit(determine_plan_type(price_id))
        }

class DatabaseSubscriptionProvider:
    """データベースからサブスクリプション情報を取得するクラス"""
    
    def __init__(self, conn):
        self.conn = conn
    
    def get_user_id_from_customer(self, customer_id: str) -> Optional[str]:
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT user_id FROM stripe_customers WHERE customer_id = %s",
            (customer_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result['user_id'] if result else None
    
    def fetch_stripe_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("""
            SELECT ss.*, s.user_id, s.plan_type, s.status as app_status, s.monthly_token_limit
            FROM stripe_subscriptions ss
            LEFT JOIN subscriptions s ON ss.subscription_id = s.stripe_subscription_id
        """)
        result = cursor.fetchall()
        cursor.close()
        
        return {sub["subscription_id"]: dict(sub) for sub in result}
    
    def fetch_app_subscriptions(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("""
            SELECT s.*, ss.price_id, ss.status as stripe_status, ss.current_period_end, ss.cancel_at_period_end
            FROM subscriptions s
            LEFT JOIN stripe_subscriptions ss ON s.stripe_subscription_id = ss.subscription_id
        """)
        result = cursor.fetchall()
        cursor.close()
        
        return [dict(sub) for sub in result]

class SubscriptionConsistencyChecker:
    """Stripeとデータベースの整合性をチェックするクラス（読み取り専用）"""
    
    def __init__(self, conn):
        self.conn = conn
        self.stripe_provider = StripeSubscriptionProvider()
        self.db_provider = DatabaseSubscriptionProvider(conn)
    
    def check_discrepancies(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        データベースとStripeの整合性をチェックして、不一致を検出する
        
        Returns:
            Dict containing:
            - missing_in_db: Stripeに存在するがDBに存在しないサブスクリプション
            - missing_in_stripe: DBに存在するがStripeに存在しないサブスクリプション 
            - mismatched: 両方に存在するがデータが一致しないサブスクリプション
        """
        logger.info("整合性チェックを開始しています...")
        
        # Stripeとデータベースからデータを取得
        stripe_subscriptions = self.stripe_provider.fetch_all_subscriptions()
        db_stripe_subscriptions = self.db_provider.fetch_stripe_subscriptions()
        
        # 結果を格納する辞書
        discrepancies = {
            "missing_in_db": [],
            "missing_in_stripe": [],
            "mismatched": []
        }
        
        stripe_subs_by_id = {sub["id"]: sub for sub in stripe_subscriptions}
        
        # 1. Stripeにあるが、DBにないサブスクリプションを確認
        for sub_id, stripe_sub in stripe_subs_by_id.items():
            if sub_id not in db_stripe_subscriptions:
                customer_id = stripe_sub["customer"]
                user_id = self.db_provider.get_user_id_from_customer(customer_id)
                
                formatted_sub = self.stripe_provider.format_subscription_data(stripe_sub, user_id)
                discrepancies["missing_in_db"].append(formatted_sub)
        
        # 2. DBにあるが、Stripeにないサブスクリプションを確認
        for db_sub_id, db_sub in db_stripe_subscriptions.items():
            if db_sub_id not in stripe_subs_by_id:
                discrepancies["missing_in_stripe"].append({
                    "subscription_id": db_sub_id,
                    "user_id": db_sub["user_id"],
                    "customer_id": db_sub["customer_id"],
                    "status": db_sub["status"],
                    "price_id": db_sub["price_id"]
                })
        
        # 3. 両方に存在するが、データが一致しないサブスクリプションを確認
        for sub_id, stripe_sub in stripe_subs_by_id.items():
            if sub_id in db_stripe_subscriptions:
                db_sub = db_stripe_subscriptions[sub_id]
                formatted_stripe_sub = self.stripe_provider.format_subscription_data(stripe_sub, db_sub["user_id"])
                
                field_discrepancies = []
                
                # ステータスを確認
                if db_sub["status"] != formatted_stripe_sub["app_status"]:
                    field_discrepancies.append({
                        "field": "stripe_subscription.status",
                        "db_value": db_sub["status"],
                        "stripe_value": formatted_stripe_sub["app_status"]
                    })
                
                # 価格IDを確認
                if db_sub["price_id"] != formatted_stripe_sub["price_id"]:
                    field_discrepancies.append({
                        "field": "price_id",
                        "db_value": db_sub["price_id"],
                        "stripe_value": formatted_stripe_sub["price_id"]
                    })
                
                # 現在の期間終了日を確認
                if db_sub.get("current_period_end"):
                    db_period_end = db_sub["current_period_end"]
                    stripe_period_end = formatted_stripe_sub["current_period_end"]
                    if db_period_end.date() != stripe_period_end.date():
                        field_discrepancies.append({
                            "field": "current_period_end",
                            "db_value": db_period_end.isoformat(),
                            "stripe_value": stripe_period_end.isoformat()
                        })
                
                # cancel_at_period_endを確認
                if db_sub["cancel_at_period_end"] != formatted_stripe_sub["cancel_at_period_end"]:
                    field_discrepancies.append({
                        "field": "cancel_at_period_end",
                        "db_value": db_sub["cancel_at_period_end"],
                        "stripe_value": formatted_stripe_sub["cancel_at_period_end"]
                    })
                
                # アプリのサブスクリプションがある場合は、そのデータも確認
                if db_sub["user_id"]:
                    # プランタイプを確認
                    if db_sub["plan_type"] != formatted_stripe_sub["plan_type"]:
                        field_discrepancies.append({
                            "field": "subscription.plan_type",
                            "db_value": db_sub["plan_type"],
                            "stripe_value": formatted_stripe_sub["plan_type"]
                        })
                    
                    # ステータスを確認
                    if db_sub["app_status"] != formatted_stripe_sub["app_status"]:
                        field_discrepancies.append({
                            "field": "subscription.status",
                            "db_value": db_sub["app_status"],
                            "stripe_value": formatted_stripe_sub["app_status"]
                        })
                    
                    # 月間トークン上限を確認
                    if db_sub["monthly_token_limit"] != formatted_stripe_sub["monthly_token_limit"]:
                        field_discrepancies.append({
                            "field": "monthly_token_limit",
                            "db_value": db_sub["monthly_token_limit"],
                            "stripe_value": formatted_stripe_sub["monthly_token_limit"]
                        })
                
                if field_discrepancies:
                    discrepancies["mismatched"].append({
                        "subscription_id": sub_id,
                        "customer_id": db_sub["customer_id"],
                        "user_id": db_sub["user_id"],
                        "discrepancies": field_discrepancies,
                        "stripe_data": formatted_stripe_sub
                    })
        
        return discrepancies
    
    def display_results(self, discrepancies: Dict[str, List[Dict[str, Any]]]) -> None:
        """整合性チェックの結果を詳細に表示"""
        missing_in_db = discrepancies["missing_in_db"]
        missing_in_stripe = discrepancies["missing_in_stripe"]
        mismatched = discrepancies["mismatched"]
        
        print("\n" + "="*80)
        print("データベース・Stripe整合性チェック結果")
        print("="*80)
        
        # 統計情報
        total_issues = len(missing_in_db) + len(missing_in_stripe) + len(mismatched)
        print(f"発見された問題の総数: {total_issues}件")
        print()
        
        # 1. Stripeにあるが、DBにないサブスクリプション
        print(f"【1】Stripeに存在するがデータベースに存在しないサブスクリプション: {len(missing_in_db)}件")
        if missing_in_db:
            for i, sub in enumerate(missing_in_db, 1):
                print(f"  {i}. サブスクリプションID: {sub['stripe_subscription_id']}")
                print(f"     カスタマーID: {sub['customer_id']}")
                print(f"     ユーザーID: {sub['user_id'] or '不明'}")
                print(f"     プランタイプ: {sub['plan_type']}")
                print(f"     ステータス: {sub['app_status']}")
                print(f"     価格ID: {sub['price_id']}")
                print(f"     期間終了日: {sub['current_period_end']}")
                print()
        else:
            print("  なし")
            print()
        
        # 2. DBにあるが、Stripeにないサブスクリプション
        print(f"【2】データベースに存在するがStripeに存在しないサブスクリプション: {len(missing_in_stripe)}件")
        if missing_in_stripe:
            for i, sub in enumerate(missing_in_stripe, 1):
                print(f"  {i}. サブスクリプションID: {sub['subscription_id']}")
                print(f"     カスタマーID: {sub['customer_id']}")
                print(f"     ユーザーID: {sub['user_id'] or '不明'}")
                print(f"     ステータス: {sub['status']}")
                print(f"     価格ID: {sub['price_id']}")
                print()
        else:
            print("  なし")
            print()
        
        # 3. 両方に存在するが、データが一致しないサブスクリプション
        print(f"【3】データが一致しないサブスクリプション: {len(mismatched)}件")
        if mismatched:
            for i, sub in enumerate(mismatched, 1):
                print(f"  {i}. サブスクリプションID: {sub['subscription_id']}")
                print(f"     カスタマーID: {sub['customer_id']}")
                print(f"     ユーザーID: {sub['user_id'] or '不明'}")
                print(f"     不一致フィールド:")
                for disc in sub["discrepancies"]:
                    print(f"     - {disc['field']}:")
                    print(f"       データベース値: {disc['db_value']}")
                    print(f"       Stripe値: {disc['stripe_value']}")
                print()
        else:
            print("  なし")
            print()
        
        print("="*80)
        
        # 対応が必要な場合のメッセージ
        if total_issues > 0:
            print(f"合計 {total_issues}件の不整合が検出されました。")
            print("手動での対応または sync_stripe_subscription.py スクリプトでの修正を検討してください。")
        else:
            print("すべてのデータが整合性を保っています。")
        
        print("="*80)
        print()

def main():
    """メイン関数"""
    conn = None
    try:
        # データベースに接続
        conn = psycopg2.connect(**DB_PARAMS)
        logger.info("データベースに接続しました")
        
        # 整合性チェッカーを初期化
        checker = SubscriptionConsistencyChecker(conn)
        
        # 整合性チェックを実行
        discrepancies = checker.check_discrepancies()
        
        # 結果を表示
        checker.display_results(discrepancies)
        
        # CSV出力のオプション（必要に応じてコメントアウト）
        # export_to_csv(discrepancies)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise
    
    finally:
        if conn:
            conn.close()
            logger.info("データベース接続を閉じました")

def export_to_csv(discrepancies: Dict[str, List[Dict[str, Any]]]) -> None:
    """結果をCSVファイルに出力（オプション機能）"""
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"subscription_consistency_check_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['問題タイプ', 'サブスクリプションID', 'カスタマーID', 'ユーザーID', 
                     'フィールド', 'データベース値', 'Stripe値', '詳細']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 各問題タイプについて書き込み
        for sub in discrepancies["missing_in_db"]:
            writer.writerow({
                '問題タイプ': 'DBに存在しない',
                'サブスクリプションID': sub['stripe_subscription_id'],
                'カスタマーID': sub['customer_id'],
                'ユーザーID': sub['user_id'] or '',
                'フィールド': '',
                'データベース値': '',
                'Stripe値': '',
                '詳細': f"プラン: {sub['plan_type']}, ステータス: {sub['app_status']}"
            })
        
        for sub in discrepancies["missing_in_stripe"]:
            writer.writerow({
                '問題タイプ': 'Stripeに存在しない',
                'サブスクリプションID': sub['subscription_id'],
                'カスタマーID': sub['customer_id'],
                'ユーザーID': sub['user_id'] or '',
                'フィールド': '',
                'データベース値': '',
                'Stripe値': '',
                '詳細': f"ステータス: {sub['status']}, 価格ID: {sub['price_id']}"
            })
        
        for sub in discrepancies["mismatched"]:
            for disc in sub["discrepancies"]:
                writer.writerow({
                    '問題タイプ': 'データ不一致',
                    'サブスクリプションID': sub['subscription_id'],
                    'カスタマーID': sub['customer_id'],
                    'ユーザーID': sub['user_id'] or '',
                    'フィールド': disc['field'],
                    'データベース値': str(disc['db_value']),
                    'Stripe値': str(disc['stripe_value']),
                    '詳細': ''
                })
    
    logger.info(f"結果をCSVファイルに出力しました: {filename}")

if __name__ == "__main__":
    main()