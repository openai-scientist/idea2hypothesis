"""Error type of the ``/v1`` routes: every error body is ``{"code", "message"}``."""

from __future__ import annotations

from typing import Any


class ContractError(Exception):
    """An error with an HTTP status and a stable wire code."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra

    def body(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}
