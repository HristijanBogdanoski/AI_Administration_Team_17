from app.core.config import settings
import google.generativeai as genai

# Configure once at import time
genai.configure(api_key=settings.gemini_api_key)

FALLBACK_TEXT = "Извини, немам доволно информации за таа услуга. Те молам прашај за друга административна услуга."

SYSTEM_PROMPT = """
Ти си виртуелен асистент за јавни административни услуги во Северна Македонија.
Одговарај кратко и јасно.

- СЕКОГАШ одговарај на МАКЕДОНСКИ јазик (кирилица).
- Ако недостигаат официјални информации, напиши "Не е наведено" наместо да измислуваш.
- Не претпоставувај институции од други држави.

Секогаш врати ОВАЈ формат (точно овие полиња):
Service Name: <име на услугата на македонски>
Required Documents: <кратка листа на документи на македонски; ако не знаеш -> "Не е наведено">
Location Data: <каде се поднесува барањето во Северна Македонија; ако не знаеш -> "Не е наведено">

Ако прашањето не е административна услуга или не можеш да препознаеш која услуга е,
врати ТОЧНО:
FALLBACK
""".strip()

def ask_gemini(question: str) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = genai.GenerativeModel(settings.gemini_model)

    prompt = f"""{SYSTEM_PROMPT}

Прашање: {question}
"""
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", None)
    return (text or "").strip()