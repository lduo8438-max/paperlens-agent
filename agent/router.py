import re


SUMMARY_PATTERNS = re.compile(
    r"\b(summarize|summary|overview|abstract|what is this paper|tldr)\b", re.I
)
SECTION_PATTERNS = re.compile(
    r"\b(section|introduction|method|result|conclusion|related work)\b", re.I
)


def route(query: str) -> str:
    """Return 'summary', 'section', or 'qa'."""
    q = query[:500]  # cap length to avoid regex backtracking
    if SUMMARY_PATTERNS.search(q):
        return "summary"
    if SECTION_PATTERNS.search(q):
        return "section"
    return "qa"
