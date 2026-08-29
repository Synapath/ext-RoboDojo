from __future__ import annotations

from collections.abc import Iterable


class LayoutSelectionError(ValueError):
    """Raised when an explicit evaluation layout selection is invalid."""


def select_layout_ids(raw: str | None, available: Iterable[int]) -> list[int]:
    available_ids = [int(value) for value in available]
    if raw is None or not raw.strip():
        return available_ids
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise LayoutSelectionError("ROBODOJO_LAYOUT_IDS must be comma-separated integers")
    try:
        requested = [int(token) for token in tokens]
    except ValueError as error:
        raise LayoutSelectionError(
            "ROBODOJO_LAYOUT_IDS must be comma-separated integers"
        ) from error
    if any(value < 0 for value in requested):
        raise LayoutSelectionError("layout ids must be non-negative")
    if len(requested) != len(set(requested)):
        raise LayoutSelectionError("layout ids must be unique")
    unknown = sorted(set(requested) - set(available_ids))
    if unknown:
        raise LayoutSelectionError(f"requested layouts are unavailable: {unknown}")
    return requested
