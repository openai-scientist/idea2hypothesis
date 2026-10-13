"""Contract tests run with the service key configured."""

import pytest

from tests.fixtures.api_helpers import SERVER_KEY


@pytest.fixture(autouse=True)
def service_key(monkeypatch):
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)
