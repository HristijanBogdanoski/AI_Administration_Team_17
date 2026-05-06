from app.core.config import settings
import google.generativeai as genai

# Configure once at import time
genai.configure(api_key=settings.gemini_api_key)

FALLBACK_TEXT = "Извини, немам доволно информации за таа услуга. Те молам прашај за друга административна услуга."

SYSTEM_PROMPT = """
Ти си виртуелен асистент за јавни административни услуги во Северна Македонија.
Пиши природно, разговорно и професионално.

- СЕКОГАШ одговарај на МАКЕДОНСКИ јазик (кирилица).
- Ако недостигаат официјални информации, јасно кажи "Не е наведено" наместо да измислуваш.
- Не претпоставувај институции од други држави.
- Одговарај со краток вовед и потоа најважните чекори/документи во природни реченици.
- Користи булети само ако навистина помагаат за читливост.
- Ако има веб-извори, вметни кратко повикување во текст (пример: "според mvr.gov.mk").

Ако прашањето не е административна услуга или не можеш да препознаеш која услуга е,
врати ТОЧНО:
FALLBACK
""".strip()

def ask_gemini(question: str, web_context: str = "") -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = genai.GenerativeModel(settings.gemini_model)

    context_block = (
        f"\nВеб-контекст (користи само релевантни и веродостојни извори):\n{web_context}\n"
        if web_context
        else ""
    )

    prompt = f"""{SYSTEM_PROMPT}
{context_block}
Прашање: {question}
"""
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", None)
    return (text or "").strip()