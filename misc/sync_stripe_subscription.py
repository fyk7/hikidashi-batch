import os
import stripe
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from datetime import datetime
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
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
    """Stripe APIからサブスクリプション情報を取得するクラス
    
    このクラスはStripeのデータを「正」として扱い、アプリケーションのDBはこれに合わせて更新される。
    Webhookの間に発生する可能性のある不整合を修正するために使用される。
    """
    
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
    def get_subscription_by_id(subscription_id: str) -> Dict[str, Any]:
        """特定のサブスクリプションをIDで取得"""
        return stripe.Subscription.retrieve(subscription_id)
    
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
    """データベースからサブスクリプション情報を取得するクラス
    
    このクラスはデータベースの読み取りと更新を担当する。
    更新時には常にStripeからのデータを正として扱う。
    """
    
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
    
    def save_stripe_subscription(self, sub_data: Dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO stripe_subscriptions (
                    subscription_id, customer_id, price_id, status, 
                    current_period_start, current_period_end, 
                    cancel_at_period_end, canceled_at,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                sub_data["stripe_subscription_id"],
                sub_data["customer_id"],
                sub_data["price_id"],
                sub_data["app_status"],
                sub_data["current_period_start"],
                sub_data["current_period_end"],
                sub_data["cancel_at_period_end"],
                sub_data["canceled_at"],
                datetime.now(),
                datetime.now()
            ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def update_stripe_subscription(self, sub_data: Dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE stripe_subscriptions
                SET price_id = %s, status = %s, current_period_start = %s, 
                    current_period_end = %s, cancel_at_period_end = %s, 
                    canceled_at = %s, updated_at = %s
                WHERE subscription_id = %s
            """, (
                sub_data["price_id"],
                sub_data["app_status"],
                sub_data["current_period_start"],
                sub_data["current_period_end"],
                sub_data["cancel_at_period_end"],
                sub_data["canceled_at"],
                datetime.now(),
                sub_data["stripe_subscription_id"]
            ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def save_or_update_app_subscription(self, sub_data: Dict[str, Any]) -> None:
        if not sub_data["user_id"]:
            return
        
        cursor = self.conn.cursor()
        try:
            # 既存のSubscriptionを確認
            cursor.execute("""
                SELECT id FROM subscriptions WHERE user_id = %s
            """, (sub_data["user_id"],))
            
            subscription_exists = cursor.fetchone() is not None
            
            if subscription_exists:
                # Subscriptionを更新
                cursor.execute("""
                    UPDATE subscriptions 
                    SET plan_type = %s, status = %s, monthly_token_limit = %s, 
                        stripe_subscription_id = %s, updated_at = %s
                    WHERE user_id = %s
                """, (
                    sub_data["plan_type"],
                    sub_data["app_status"],
                    sub_data["monthly_token_limit"],
                    sub_data["stripe_subscription_id"],
                    datetime.now(),
                    sub_data["user_id"]
                ))
            else:
                # Subscriptionを作成
                cursor.execute("""
                    INSERT INTO subscriptions (
                        id, user_id, plan_type, status, monthly_token_limit,
                        stripe_subscription_id, created_at, updated_at
                    ) VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s)
                """, (
                    sub_data["user_id"],
                    sub_data["plan_type"],
                    sub_data["app_status"],
                    sub_data["monthly_token_limit"],
                    sub_data["stripe_subscription_id"],
                    datetime.now(),
                    datetime.now()
                ))
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()

class SubscriptionSynchronizer:
    """StripeとDBのサブスクリプション情報を比較し同期するクラス
    
    常にStripeのデータを「正」として扱い、DBのデータはStripeに合わせて更新される。
    """
    
    def __init__(self, conn):
        self.conn = conn
        self.stripe_provider = StripeSubscriptionProvider()
        self.db_provider = DatabaseSubscriptionProvider(conn)
    
    def check_discrepancies(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """サブスクリプションの不一致を確認
        
        Stripeを正として、DBとの不一致を検出する。
        """
        stripe_subscriptions = self.stripe_provider.fetch_all_subscriptions()
        db_stripe_subscriptions = self.db_provider.fetch_stripe_subscriptions()
        missing_in_db = []
        mismatched = []
        
        stripe_subs_by_id = {sub["id"]: sub for sub in stripe_subscriptions}
        
        # 1. Stripeにあるが、DBにないサブスクリプションを確認
        for sub_id, stripe_sub in stripe_subs_by_id.items():
            if sub_id not in db_stripe_subscriptions:
                # カスタマーからユーザーIDを取得
                customer_id = stripe_sub["customer"]
                user_id = self.db_provider.get_user_id_from_customer(customer_id)
                
                missing_in_db.append(self.stripe_provider.format_subscription_data(stripe_sub, user_id))
        
        # 3. 存在するが、データが一致しないサブスクリプションを確認
        for sub_id, stripe_sub in stripe_subs_by_id.items():
            if sub_id in db_stripe_subscriptions:
                db_sub = db_stripe_subscriptions[sub_id]
                formatted_stripe_sub = self.stripe_provider.format_subscription_data(stripe_sub, db_sub["user_id"])
                
                discrepancies = []
                
                # ステータスを確認
                if db_sub["status"] != formatted_stripe_sub["app_status"]:
                    discrepancies.append({
                        "field": "stripe_subscription.status",
                        "db_value": db_sub["status"],
                        "stripe_value": formatted_stripe_sub["app_status"]
                    })
                
                # 価格IDを確認
                if db_sub["price_id"] != formatted_stripe_sub["price_id"]:
                    discrepancies.append({
                        "field": "price_id",
                        "db_value": db_sub["price_id"],
                        "stripe_value": formatted_stripe_sub["price_id"]
                    })
                
                # 現在の期間終了日を確認
                db_period_end = db_sub["current_period_end"]
                stripe_period_end = formatted_stripe_sub["current_period_end"]
                if db_period_end.date() != stripe_period_end.date():
                    discrepancies.append({
                        "field": "current_period_end",
                        "db_value": db_period_end.isoformat(),
                        "stripe_value": stripe_period_end.isoformat()
                    })
                
                # cancel_at_period_endを確認
                if db_sub["cancel_at_period_end"] != formatted_stripe_sub["cancel_at_period_end"]:
                    discrepancies.append({
                        "field": "cancel_at_period_end",
                        "db_value": db_sub["cancel_at_period_end"],
                        "stripe_value": formatted_stripe_sub["cancel_at_period_end"]
                    })
                
                # アプリのサブスクリプションがある場合は、そのデータも確認
                if db_sub["user_id"]:
                    # プランタイプを確認
                    if db_sub["plan_type"] != formatted_stripe_sub["plan_type"]:
                        discrepancies.append({
                            "field": "subscription.plan_type",
                            "db_value": db_sub["plan_type"],
                            "stripe_value": formatted_stripe_sub["plan_type"]
                        })
                    
                    # ステータスを確認
                    if db_sub["app_status"] != formatted_stripe_sub["app_status"]:
                        discrepancies.append({
                            "field": "subscription.status",
                            "db_value": db_sub["app_status"],
                            "stripe_value": formatted_stripe_sub["app_status"]
                        })
                    
                    # 月間トークン上限を確認
                    if db_sub["monthly_token_limit"] != formatted_stripe_sub["monthly_token_limit"]:
                        discrepancies.append({
                            "field": "monthly_token_limit",
                            "db_value": db_sub["monthly_token_limit"],
                            "stripe_value": formatted_stripe_sub["monthly_token_limit"]
                        })
                
                if discrepancies:
                    mismatched.append({
                        "subscription_id": sub_id,
                        "customer_id": db_sub["customer_id"],
                        "user_id": db_sub["user_id"],
                        "discrepancies": discrepancies,
                        "stripe_data": formatted_stripe_sub
                    })
        
        # return missing_in_db, missing_in_stripe, mismatched
        return missing_in_db, mismatched
    
    def fix_discrepancies(self, missing_in_db: List[Dict[str, Any]], mismatched: List[Dict[str, Any]]) -> None:
        """不一致を修正
        
        Stripeのデータを正として、DBのデータを更新する。
        1. Stripeにあって、DBにないサブスクリプションはDBに追加
        2. 両方に存在するが、データが一致しないサブスクリプションはStripeのデータでDBを更新
        """
        # 1. Stripeにあるが、DBにないサブスクリプションを修正
        for sub in missing_in_db:
            if not sub["user_id"]:
                logger.warning(f"警告: ユーザーIDが見つからないため、サブスクリプション {sub['stripe_subscription_id']} を追加できません")
                continue
            
            try:
                self.db_provider.save_stripe_subscription(sub)
                self.db_provider.save_or_update_app_subscription(sub)
                logger.info(f"データベースに追加されたサブスクリプション: {sub['stripe_subscription_id']}")
            
            except Exception as e:
                logger.error(f"エラー: サブスクリプション {sub['stripe_subscription_id']} の追加中に問題が発生しました: {str(e)}")
        
        # 3. 存在するが、データが一致しないサブスクリプションを修正
        for sub in mismatched:
            try:
                stripe_data = sub["stripe_data"]
                self.db_provider.update_stripe_subscription(stripe_data)
                if sub["user_id"]:
                    self.db_provider.save_or_update_app_subscription(stripe_data)
                
                logger.info(f"更新されたサブスクリプション: {sub['subscription_id']}")
            
            except Exception as e:
                logger.error(f"エラー: サブスクリプション {sub['subscription_id']} の更新中に問題が発生しました: {str(e)}")
  
    def display_results(self, missing_in_db: List[Dict[str, Any]], mismatched: List[Dict[str, Any]]) -> None:
        """結果を表示"""
        logger.info("=== 同期状態の確認結果 ===")
        logger.info(f"1. Stripeにあるが、DBにないサブスクリプション: {len(missing_in_db)}件")
        for i, sub in enumerate(missing_in_db, 1):
            logger.info(f"  {i}. サブスクリプションID: {sub['stripe_subscription_id']}, ユーザーID: {sub['user_id'] or '不明'}")
     
        logger.info(f"2. 存在するが、データが一致しないサブスクリプション: {len(mismatched)}件")
        for i, sub in enumerate(mismatched, 1):
            logger.info(f"  {i}. サブスクリプションID: {sub['subscription_id']}, ユーザーID: {sub['user_id'] or '不明'}")
            logger.info("     不一致のフィールド:")
            for disc in sub["discrepancies"]:
                logger.info(f"     - {disc['field']}: DB={disc['db_value']}, Stripe={disc['stripe_value']}")

# メイン関数
def main():
    # データベースに接続
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        
        logger.info("Stripeサブスクリプションとデータベースの同期状態を確認しています...")
        
        # 同期処理を実行
        synchronizer = SubscriptionSynchronizer(conn)
        # missing_in_db, missing_in_stripe, mismatched = synchronizer.check_discrepancies()
        missing_in_db, mismatched = synchronizer.check_discrepancies()
        
        # 結果の表示
        # synchronizer.display_results(missing_in_db, missing_in_stripe, mismatched)
        synchronizer.display_results(missing_in_db, mismatched)
        
        # 修正するかどうかの確認
        # if missing_in_db or missing_in_stripe or mismatched:
        if missing_in_db or mismatched:
            confirm = input("\n不一致を修正しますか？ (y/n): ")
            if confirm.lower() == 'y':
                logger.info("不一致を修正しています...")
                # synchronizer.fix_discrepancies(missing_in_db, missing_in_stripe, mismatched)
                synchronizer.fix_discrepancies(missing_in_db, mismatched)
                logger.info("修正が完了しました。")
                
                # 以下の部分はコメントアウト - 自前のDBに存在してStripeに存在しないケースは起こり得ないため
                """
                # FREEプランに変更されたユーザーのトークン使用量をリセット
                synchronizer.reset_token_usage_for_downgraded_users(missing_in_stripe)
                """
            else:
                logger.info("修正はキャンセルされました。")
        else:
            logger.info("すべてのサブスクリプションは同期されています。修正は必要ありません。")
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
