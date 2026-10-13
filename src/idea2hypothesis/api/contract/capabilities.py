"""Declarations for the i2h engine's published API."""

from idea2hypothesis import __version__
from idea2hypothesis.api.contract.models import (
    CONTRACT_VERSION,
    Capabilities,
    OutputDecl,
    ResearchProfile,
    StepInfo,
)
from idea2hypothesis.pipeline.models import Stage

ENGINE_KEY = "i2h"
STEP_KEYS = tuple(stage.name.lower() for stage in Stage)


def capabilities() -> Capabilities:
    return Capabilities(
        engine_key=ENGINE_KEY,
        engine_version=__version__,
        contract_version=CONTRACT_VERSION,
        stage_keys=["hypothesis"],
        output_schemas=[
            OutputDecl(schema="i2h.hypotheses/v1", primary=True),
            OutputDecl(schema="review.findings/v1"),
            OutputDecl(schema="revision.response/v1"),
            OutputDecl(schema="revision.diff/v1"),
            OutputDecl(schema="i2h.argument_map/v1"),
        ],
        steps=[StepInfo(step_key=key, label=key.replace("_", " "), order=i)
               for i, key in enumerate(STEP_KEYS, 1)],
        question_gates={
            "copilot": ["literature_screen"],
            "full": ["problem_decompose", "literature_screen"],
        },
        optional_integrations=["semantic_scholar", "openalex"],
        providers=["openai_compatible", "bedrock"],
        auto_revision=False,
        profiles=[ResearchProfile()],
    )
