from __future__ import annotations

from idea2hypothesis.stages.base import compact_json


def test_compact_json_has_no_padding_and_keeps_unicode() -> None:
    assert compact_json({"a": [1, 2], "b": "é"}) == '{"a":[1,2],"b":"é"}'
