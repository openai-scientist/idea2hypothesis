"""Test doubles. Everything here is a clearly named fixture; the package never imports it."""

from tests.fixtures.fixture_literature import FixtureLiterature, make_fixture_papers
from tests.fixtures.fixture_llm import FixtureLLM, PromptInfo
from tests.fixtures.fixture_novelty import FixtureNoveltyJudge

__all__ = [
    "FixtureLLM",
    "FixtureLiterature",
    "FixtureNoveltyJudge",
    "PromptInfo",
    "make_fixture_papers",
]
