import os
import httpx
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
COOLDOWN_SECONDS = 30

class TelegramAlertService:
    def __init__(self):
        self.last_alert_time = None
        self.client = httpx.AsyncClient()

    async def send_alert(self, image_path: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Telegram credentials not set. Skipping alert.")
            return

        now = datetime.now()
        
        # Check cooldown
        if self.last_alert_time and (now - self.last_alert_time).total_seconds() < COOLDOWN_SECONDS:
            print("Alert in cooldown. Skipping.")
            return
            
        self.last_alert_time = now

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        time_str = now.strftime("%H:%M:%S")
        caption = f"⚠️ Внимание! Обнаружено падение.\nВремя: {time_str}"

        try:
            with open(image_path, "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
                response = await self.client.post(url, data=data, files=files)
                response.raise_for_status()
                print("Telegram alert sent successfully.")
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
            
    async def close(self):
        await self.client.aclose()

telegram_service = TelegramAlertService()
