from tavily import TavilyClient

from app.core.config import settings


def _build_context_from_results(payload: dict) -> str:
    answer = (payload.get("answer") or "").strip()
    results = payload.get("results") or []

    lines: list[str] = []

    if answer:
        lines.append(f"Summary: {answer}")

    for index, item in enumerate(results, start=1):
        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if len(content) > 500:
            content = f"{content[:500]}..."

        lines.append(f"Source {index}: {title}")
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(f"Content: {content}")

    return "\n".join(lines).strip()


def search_context(question: str) -> str:
    if not settings.tavily_api_key:
        return ""

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        payload = client.search(
            query=question,
            search_depth=settings.tavily_search_depth,
            max_results=settings.tavily_max_results,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception:
        return ""

    return _build_context_from_results(payload)

