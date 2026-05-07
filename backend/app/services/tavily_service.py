from tavily import TavilyClient
from urllib.parse import urlparse

from app.core.config import settings


def _is_allowed_domain(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False

    if not allowed_domains:
        return host.endswith(".mk") or host == "mk"

    for allowed in allowed_domains:
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _filter_results(results: list[dict], allowed_domains: list[str]) -> list[dict]:
    return [item for item in results if _is_allowed_domain((item.get("url") or "").strip(), allowed_domains)]


def _build_context_from_results(payload: dict, allowed_domains: list[str]) -> str:
    answer = (payload.get("answer") or "").strip()
    results = _filter_results(payload.get("results") or [], allowed_domains)

    lines: list[str] = []

    if answer and results:
        lines.append(f"Краток веб-преглед: {answer}")

    for index, item in enumerate(results, start=1):
        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip()
        domain = (urlparse(url).hostname or "").strip()
        content = (item.get("content") or "").strip()
        if len(content) > 420:
            content = f"{content[:420]}..."

        lines.append(f"Извор {index}: {title}")
        if domain:
            lines.append(f"Домен: {domain}")
        if url:
            lines.append(f"Линк: {url}")
        if content:
            lines.append(f"Релевантен извадок: {content}")

    return "\n".join(lines).strip()


def search_context(question: str) -> str:
    if not settings.tavily_api_key:
        return ""

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        allowed_domains = settings.tavily_include_domains
        scoped_question = f"{question} Северна Македонија јавна администрација"
        payload = client.search(
            query=scoped_question,
            search_depth=settings.tavily_search_depth,
            max_results=settings.tavily_max_results,
            include_answer=True,
            include_raw_content=False,
            include_domains=allowed_domains,
        )
    except Exception:
        return ""

    context = _build_context_from_results(payload, allowed_domains)
    if context:
        return context

    if settings.tavily_strict_sources:
        return ""

    # Optional non-strict fallback if no whitelisted source matched.
    return _build_context_from_results(payload, [])

