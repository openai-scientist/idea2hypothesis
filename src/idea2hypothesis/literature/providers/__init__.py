"""Literature providers: OpenAlex, Semantic Scholar and arXiv."""

from idea2hypothesis.literature.providers.arxiv import ArxivProvider
from idea2hypothesis.literature.providers.http import CircuitBreaker, ProviderError
from idea2hypothesis.literature.providers.openalex import OpenAlexProvider
from idea2hypothesis.literature.providers.semantic_scholar import SemanticScholarProvider

__all__ = [
    "ArxivProvider",
    "CircuitBreaker",
    "OpenAlexProvider",
    "ProviderError",
    "SemanticScholarProvider",
]
