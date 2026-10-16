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
    if review_doc.get("error") and not any(f.severity == "blocking" for f in findings):
        # The review itself failed: say so as a blocking finding on the whole set (as exd does),
        # so `passed: false` is valid for the Platform and the gap travels as a known risk.
        findings.append(
            Finding(
                finding_id=f"F{round_}-{len(findings) + 1}",
                anchor=Anchor(locator="document:hypotheses", artifact_id=artifact_id),
                severity="blocking",
                issue="Review incomplete: the reviewer returned no valid output "
                f"({review_doc['error']}). The hypotheses were not fully inspected.",
                author="reviewer",
                kind="needs_evidence",
                suggestion="Treat this review as partial; repeat it if the set is still in doubt.",
            )
        )
    return ReviewFindings(
        round=round_,
        reviewed_artifact_id=artifact_id,
        reviewer_model=str(review_doc["reviewer_model"]),
        findings=findings,
        passed=bool(review_doc["passed"]),
    )


def _no_reason(outcome: str) -> str | None:
    """The Platform requires a note unless the item is fixed; the reviser may leave it out."""
    return None if outcome == "fixed" else "No reason given by the reviser."


def _note(note: Any, outcome: str) -> str | None:
    return note if isinstance(note, str) and note.strip() else _no_reason(outcome)


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
                note=_note(item.get("note"), item["outcome"]),
            )
            for item in responses_doc["items"]
        ],
    )
