from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name":                 {"type": "string"},
        "category":             {"type": "string"},
        "estimated_price":      {"type": "string"},
        "is_real_world_item":   {"type": "boolean"},
        "packaging_form":       {"type": "string"},
        "accent_color":         {"type": "string"},
        "visual_style":         {"type": "string"},
        "role_in_box":          {"type": "string"},
        "description":          {"type": "string"},
        "reason":               {"type": "string"},
        "emotional_trigger":    {"type": "string"},
        "personalization_detail": {"type": "string"},
        "search_query":         {"type": "string"},
    },
    "required": [
        "name", "category", "estimated_price", "is_real_world_item",
        "packaging_form", "accent_color", "visual_style", "role_in_box",
        "description", "reason", "emotional_trigger",
        "personalization_detail", "search_query",
    ],
}

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "archetype":             {"type": "string"},
        "box_narrative":         {"type": "string"},
        "shape":                 {"type": "string", "enum": ["square", "rectangular", "circular"]},
        "color":                 {"type": "string"},
        "interior_accent_colors": {"type": "array", "items": {"type": "string"}},
        "filler":                {"type": "string"},
        "theme":                 {"type": "string"},
        "unboxing_order":        {"type": "array", "items": {"type": "string"}},
        "items":                 {"type": "array", "items": ITEM_SCHEMA},
        "lighting":              {"type": "string"},
    },
    "required": [
        "archetype", "box_narrative", "shape", "color",
        "interior_accent_colors", "filler", "theme",
        "unboxing_order", "items", "lighting",
    ],
}


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in model response.")

def safe_parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    extracted = extract_json_block(cleaned)
    return json.loads(extracted)


def classify_giftbox_request(user_input: str) -> tuple[bool, str]:
    """
    İstifadəçinin mesajını təhlil edir.
    Əgər mesaj hədiyyə qutusu yaratmaq üçün kifayət məlumat daşıyırsa -> (True, "")
    Əgər kifayət deyilsə və ya tamamilə fərqli mövzudadırsa -> (False, "istifadəçiyə göndəriləcək mesaj")
    """
    classification_prompt = (
        "Sən hədiyyə qutusu yaratmaq üçün istifadəçi mesajlarını analiz edən bir Aİ yardımçısısan.\n"
        "İstifadəçinin mesajını oxu və qərar ver:\n"
        "1. Bu mesaj real bir hədiyyə qutusu yaratmaq üçün **kifayət qədər məlumat** verirmi?\n"
        "   - Kifayət məlumat: şəxsin yaşı, cinsi, peşəsi, hobbi, zövqləri və ya hədiyyə ilə bağlı spesifik istəklər.\n"
        "   - Əgər mesajda sadəcə 'salam', 'necəsən', 'kömək et' kimi ümumi sözlər varsa => KİFAYƏT DEYİL.\n"
        "   - Əgər mesajda hədiyyə ilə əlaqəsiz başqa bir mövzu (hava, futbol, proqramlaşdırma) varsa => KİFAYƏT DEYİL.\n"
        "2. Əgər kifayət deyilsə, istifadəçiyə **Azərbaycan dilində** qısa bir xəbərdarlıq mesajı yaz\n"
        "   - Mesajda 'giftbox yaratmaq ucun yeterli melumat verin' ifadəsi olmalıdır.\n"
        "   - Misal: 'Bağışlayın, hədiyyə qutusu yaratmaq üçün daha ətraflı məlumat lazımdır. Zəhmət olmasa alıcının yaşı, maraqları, peşəsi və s. haqqında yazın.'\n"
        "3. Cavab FORMATI (yalnız JSON, başqa heç nə):\n"
        "   {\"is_sufficient\": boolean, \"message\": string}\n"
        "   - is_sufficient = true -> məlumat kifayətdir, message boş string olmalıdır \"\"\n"
        "   - is_sufficient = false -> message yuxarıdakı kimi xəbərdarlıq mətni olmalıdır\n"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
            contents=f"{classification_prompt}\n\nİstifadəçi mesajı: {user_input}"
        )
        raw = response.text.strip()
        # Bəzən model json-dan əvvəl/qabaq ağ boşluq qoya bilər, təmizləyək
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        is_suff = data.get("is_sufficient", False)
        msg = data.get("message", "")
        if is_suff:
            return True, ""
        else:
            # Əgər mesaj boşdursa, default xəbərdarlıq
            if not msg.strip():
                msg = "⚠️ Hədiyyə qutusu yaratmaq üçün yetərli məlumat verməmisiniz. Zəhmət olmasa alıcının yaşı, cinsi, peşəsi, hobbiləri və digər maraqlı detalları yazın."
            return False, msg
    except Exception as e:
        # Hər hansı xəta olarsa (API, parse) təhlükəsiz tərəf: istifadəçidən təkrar istə
        print(f"Classification error: {e}")
        return False, "❌ Texniki xəta: mesajınız təhlil edilə bilmədi. Zəhmət olmasa daha sonra təkrar cəhd edin."


def gift_box_startup_engine(user_input, max_retries: int = 3):
    JSON_OUTPUT_RULES = (
        "\n\n"
        "══════════════════════════════════════════════\n"
        "OUTPUT FORMAT — ABSOLUTE STRICT RULES:\n"
        "══════════════════════════════════════════════\n"
        "1. Your ENTIRE response must be a single valid JSON object.\n"
        "2. Do NOT write any text before or after the JSON.\n"
        "3. Do NOT use markdown code fences (no ```json, no ```).\n"
        "4. Do NOT add comments inside the JSON.\n"
        "5. All string values must use double quotes.\n"
        "6. Boolean values must be lowercase: true / false.\n"
        "7. The JSON must be parseable by Python's json.loads() with zero modification.\n"
        "8. If you are tempted to write anything other than JSON, DELETE IT.\n"
        "VIOLATION OF ANY RULE ABOVE MAKES THE ENTIRE RESPONSE USELESS.\n"
    )

    last_error = None
    raw_response = None
    data = None

    for attempt in range(1, max_retries + 1):
        retry_note = (
            ""
            if attempt == 1
            else (
                f"\n\nATTENTION — RETRY {attempt}/{max_retries}: "
                f"Your previous response failed JSON parsing with error: {last_error}. "
                "Fix the format. Return ONLY valid JSON. No markdown. No extra text."
            )
        )
        try:
            text_response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BOX_SCHEMA,
                    system_instruction=(
                        "You are a premium gift box curator. "
                        "Your output will be used to generate a real gift box AND a photorealistic image of it. "
                        "Both must be excellent: the box must feel deeply personal, and the image must be beautiful. "
                        "GOLDEN RULE: Every item must be instantly recognizable when seen in a photo. "
                        "If someone looks at the image and cannot immediately understand what an item is, replace it. "
                        "Items must be visually clear, real-world products that exist on Amazon, Etsy, or specialty stores. "
                        "No abstract, conceptual, or hard-to-photograph items. "
                        "STEP 1 — Read the recipient description carefully. "
                        "Identify: age, gender, profession, hobbies, personality, daily habits, lifestyle. "
                        "Write a 2-sentence box_narrative that captures who this person is and why this box fits them. "
                        "STEP 2 — Design the box shell. "
                        "Shape: choose ONE from [square, rectangular, circular]. "
                        "Color: one solid matte color that matches the recipient's personality. "
                        "Filler: natural kraft tissue or black tissue only. "
                        "STEP 3 — Select 5 to 7 items total. "
                        "RULES FOR EVERY ITEM: Must be real, purchasable, visually obvious, under $50, looks beautiful in a box. "
                        "ITEM MIX: 1 STATEMENT ITEM, 2-3 PERSONAL USE ITEMS, 1 PLAYFUL ITEM, 1 COMFORT ITEM, 1 PERSONAL TOUCH. "
                        "VISUAL RULES: different shapes/heights, at least 2 accent colors, interesting textures, avoid monochrome. "
                        "NEVER include: hard-to-identify, fantasy, generic items, or more than 7 items. "
                        + JSON_OUTPUT_RULES
                    ),
                ),
                contents=f"Recipient description: {user_input}{retry_note}"
            )
            raw_response = text_response.text.strip()
            data = safe_parse_json(raw_response)
            if not isinstance(data.get("items"), list) or len(data["items"]) == 0:
                raise ValueError("Parsed JSON has no 'items' list.")
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            print(f"[Attempt {attempt}/{max_retries}] JSON parse failed: {e}")
            if attempt == max_retries:
                print(f"[WARNING] All {max_retries} attempts failed. Using fallback data.")
                data = _fallback_data()
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

    # Items list for prompt
    items_list = []
    for item in data.get("items", []):
        if isinstance(item, dict):
            items_list.append(item.get("name", "unknown item"))
        else:
            items_list.append(str(item))

    # accent colors
    accent_colors = data.get("interior_accent_colors", ["cobalt blue", "burnt orange", "brass gold"])
    accent_colors_str = ", ".join(accent_colors)

    packaging_forms = []
    for item in data.get("items", []):
        if isinstance(item, dict) and item.get("packaging_form"):
            packaging_forms.append(item["packaging_form"])
    packaging_str = "; ".join(packaging_forms) if packaging_forms else "mixed packaging forms"

    final_prompt = (
        f"Premium boutique lifestyle brand editorial photography. "
        f"TOP-DOWN VIEW of an open {data['shape']} rigid gift box. "
        f"Box EXTERIOR: {data['color']}, perfectly matte. "
        f"INTERIOR CONTENTS: {', '.join(items_list)}. "
        f"COLOR STORY: {accent_colors_str}. "
        f"PACKAGING VARIETY: {packaging_str}. "
        f"COMPOSITION: controlled editorial chaos, asymmetric, slightly layered. "
        f"LIGHTING: {data['lighting']}, warm directional window light, golden hour. "
        f"SURFACE: warm matte wood grain or rough linen cloth. "
        f"NO logos, no brand text, no plastic shine."
    )

    image_response = client.models.generate_images(
        model="models/imagen-4.0-fast-generate-001",
        prompt=final_prompt
    )
    image = image_response.generated_images[0].image
    return image, final_prompt, data


def _fallback_data() -> dict:
    return {
        "archetype": "midnight coder who plays football on weekends",
        "box_narrative": "For the person who debugs models at 2am and still plays football on weekends.",
        "shape": "rectangular",
        "color": "matte slate grey",
        "interior_accent_colors": ["cobalt blue", "burnt orange", "brass gold", "warm cream"],
        "filler": "natural kraft tissue paper",
        "theme": "focus, field, and fuel",
        "lighting": "warm natural window light, golden hour mood",
        "unboxing_order": ["envelope", "espresso cup", "socks", "coffee vials", "notebook", "chocolate tin"],
        "items": [
            {
                "name": "wax-sealed kraft envelope",
                "category": "interactive",
                "estimated_price": "$5",
                "is_real_world_item": True,
                "packaging_form": "wax-sealed envelope",
                "accent_color": "deep burgundy",
                "visual_style": "handcrafted",
                "role_in_box": "interactive item",
                "description": "thick kraft envelope with burgundy wax seal",
                "reason": "for the 2am debugging moment",
                "emotional_trigger": "being seen",
                "personalization_detail": "initial on wax seal",
                "search_query": "wax seal envelope"
            },
            {
                "name": "cobalt blue espresso cup",
                "category": "drinkware",
                "estimated_price": "$24",
                "is_real_world_item": True,
                "packaging_form": "kraft-wrapped",
                "accent_color": "cobalt blue",
                "visual_style": "bold ceramic",
                "role_in_box": "hook item",
                "description": "hand-thrown ceramic cup",
                "reason": "specialty coffee ritual",
                "emotional_trigger": "delight",
                "personalization_detail": "none",
                "search_query": "cobalt blue ceramic espresso cup"
            },
            {
                "name": "football grip socks",
                "category": "sports",
                "estimated_price": "$18",
                "is_real_world_item": True,
                "packaging_form": "coiled with paper band",
                "accent_color": "burnt orange",
                "visual_style": "athletic",
                "role_in_box": "personality item",
                "description": "anti-slip grip socks",
                "reason": "football is his outlet",
                "emotional_trigger": "freedom",
                "personalization_detail": "printed: 'Match Day Kit'",
                "search_query": "football grip socks"
            },
            {
                "name": "specialty coffee vials",
                "category": "coffee",
                "estimated_price": "$32",
                "is_real_world_item": True,
                "packaging_form": "glass vials in wooden rack",
                "accent_color": "warm cream",
                "visual_style": "scientific",
                "role_in_box": "practical item",
                "description": "3 single-origin coffees",
                "reason": "precision and curiosity",
                "emotional_trigger": "curiosity",
                "personalization_detail": "labels: 'Sample A/B/C'",
                "search_query": "coffee sampler glass vials"
            },
            {
                "name": "coder's notebook",
                "category": "stationery",
                "estimated_price": "$22",
                "is_real_world_item": True,
                "packaging_form": "fabric-banded hardcover",
                "accent_color": "deep slate",
                "visual_style": "technical minimalist",
                "role_in_box": "practical item",
                "description": "circuit-schematic embossed cover",
                "reason": "externalizes complex problems",
                "emotional_trigger": "calm focus",
                "personalization_detail": "stamped: 'Field Notes Vol.1'",
                "search_query": "circuit notebook embossed"
            },
            {
                "name": "brass-lid chocolate tin",
                "category": "snack",
                "estimated_price": "$16",
                "is_real_world_item": True,
                "packaging_form": "metal tin",
                "accent_color": "brass gold",
                "visual_style": "artisan tactile",
                "role_in_box": "comfort item",
                "description": "dark chocolate in brass-lid tin",
                "reason": "late-night reward",
                "emotional_trigger": "warmth",
                "personalization_detail": "labeled: 'Debug Complete'",
                "search_query": "artisan dark chocolate tin"
            }
        ]
    }