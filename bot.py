import io
import json
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from gift_engine import gift_box_startup_engine, classify_giftbox_request
from PIL import Image as PILImage

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN tapılmadı. .env faylını yoxlayın.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "Salam 👋\n\n"
        "Mən sizin verdiyiniz məlumatlara əsasən **xüsusi hədiyyə 🎁 konsepti** hazırlayıram.\n\n"
        "Mənə sadəcə hədiyyə alacağınız insan haqqında məlumat verin: "
        "məsələn, cinsi 👤, yaşı 🎂, bürcü ♈, gördüyü iş (peşəsi) 💼, hobbiləri 🎯 və digər maraqlı detallar ✨."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def handle_description(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    if len(user_input) < 10:
        await update.message.reply_text("Zəhmət olmasa daha ətraflı təsvir yazın (ən azı 10 simvol).")
        return

    # 1) Mesajı təsniflə - hədiyyə qutusu üçün kifayət məlumat varmı?
    is_valid, reply_msg = classify_giftbox_request(user_input)
    if not is_valid:
        # Kifayət məlumat yoxdur və ya mövzu kənardır
        await update.message.reply_text(reply_msg)
        return

    # 2) Kifayət məlumat var, davam et
    processing_msg = await update.message.reply_text("🎁 Hədiyyə qutusu hazırlanır... Bu, 15-30 saniyə çəkə bilər.")

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        image, prompt, box_data = await loop.run_in_executor(
            None,
            gift_box_startup_engine,
            user_input
        )

        # Şəkli PIL Image-ə çevirib BytesIO-ya yaz
        img_bytes = io.BytesIO()
        
        # Google genai-dən gələn image obyektinin tipini yoxla
        if hasattr(image, 'image_bytes'):
            pil_image = PILImage.open(io.BytesIO(image.image_bytes))
        elif hasattr(image, 'data'):
            pil_image = PILImage.open(io.BytesIO(image.data))
        else:
            # Əgər birbaşa PIL Image-dirsə (nadir)
            pil_image = image
        
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_bytes.name = "gift_box.png"

        # Şəkli göndər
        await update.message.reply_photo(photo=img_bytes, caption="🖼️ Yaradılmış qutu şəkli")

        # İstifadəçi tələbinə uyğun mesaj
        order_message = (
            "🎁 Buyurun, hədiyyəniz hazırdır! 😊\n\n"
            "Bu hədiyyə qutusunu sifariş etmək üçün bu nömrə ilə əlaqə saxlayın: **0776009232**"
        )
        await update.message.reply_text(order_message, parse_mode="Markdown")

        logger.info("Şəkil və sifariş mesajı göndərildi.")
        await processing_msg.delete()

    except Exception as e:
        logger.exception("Xəta baş verdi:")
        await processing_msg.edit_text(f"❌ Xəta: {str(e)}\nZəhmət olmasa daha sonra təkrar cəhd edin.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_description))
    logger.info("Bot işə düşdü...")
    app.run_polling()

if __name__ == "__main__":
    main()