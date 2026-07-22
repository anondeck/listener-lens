from __future__ import annotations

import pytest

from earshift_bakeoff.bilingual_vowel_unseen_fixtures import (
    CONTEXT_ORDER,
    fixture_text,
)


@pytest.mark.parametrize(
    ("language", "context", "text", "indexes"),
    (
        ("en-US", CONTEXT_ORDER[0], "They see sample more now.", (2,)),
        ("en-US", CONTEXT_ORDER[1], "They say more, then sample.", (4,)),
        ("en-US", CONTEXT_ORDER[2], "They say sample, then sample.", (2, 4)),
        ("pt-BR", CONTEXT_ORDER[0], "Quem quer amostra mais?", (2,)),
        ("pt-BR", CONTEXT_ORDER[1], "Quem quer mais, diz amostra.", (4,)),
        ("pt-BR", CONTEXT_ORDER[2], "Quem diz amostra, diz amostra.", (2, 4)),
    ),
)
def test_fixture_text_has_frozen_target_positions(
    language: str,
    context: str,
    text: str,
    indexes: tuple[int, ...],
) -> None:
    assert fixture_text(
        language, context, "sample" if language == "en-US" else "amostra"
    ) == (
        text,
        indexes,
    )


def test_fixture_text_rejects_unknown_context() -> None:
    with pytest.raises(ValueError, match="unsupported unseen fixture frame"):
        fixture_text("en-US", "unknown", "sample")
