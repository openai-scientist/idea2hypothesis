"""``review.json`` to wire findings; researcher comments to revision responses."""

from __future__ import annotations

from typing import Any

from idea2hypothesis.api.contract.models import (
    Anchor,
    Finding,
    ReviewFindings,
    RevisionItem,
    RevisionResponse,
)


def review_findings(review_doc: dict[str, Any], *, artifact_id: str) -> ReviewFindings:
    """Findings of the independent review; ids are ``F<round>-<n>`` (engine-assigned)."""
    round_ = int(review_doc["round"])
    findings = [
        Finding(
            finding_id=f"F{round_}-{n}",
            anchor=Anchor(
                locator=f"hypothesis:{item['hypothesis_id']}/{item['field']}",
                artifact_id=artifact_id,
            ),
            severity=item["severity"],
            issue=item["issue"],
            author="reviewer",
            category=item["category"],
            kind=item["kind"],
            suggestion=item.get("suggestion"),
        )
        for n, item in enumerate(review_doc["findings"], 1)
    ]
    return ReviewFindings(
        round=round_,
        reviewed_artifact_id=artifact_id,
        reviewer_model=str(review_doc["reviewer_model"]),
        findings=findings,
        passed=bool(review_doc["passed"]),
    )


def revision_response(
    responses_doc: dict[str, Any], *, round_: int, before: str, after: str
) -> RevisionResponse:
    """The generator's answer to each researcher finding of the round that was revised."""
    return RevisionResponse(
        round=round_,
        artifact_before=before,
        artifact_after=after,
        items=[
            RevisionItem(
                finding_id=str(item["finding_id"]),
                outcome=item["outcome"],
                note=item.get("note"),
            )
            for item in responses_doc["items"]
        ],
    )
