"""Unit tests for the TrueLayer provider."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import pytest

from app.agents.services.crypto import decrypt
from app.providers.base import SessionExpiredError
from app.providers.truelayer import TrueLayerProvider, _token_value


@pytest.fixture
def truelayer_env(monkeypatch):
    monkeypatch.setenv("TRUELAYER_CLIENT_ID", "tl-client")
    monkeypatch.setenv("TRUELAYER_CLIENT_SECRET", "tl-secret")
    monkeypatch.setenv("TRUELAYER_AUTH_URL", "https://auth.truelayer.test")
    monkeypatch.setenv("TRUELAYER_API_URL", "https://api.truelayer.test/data/v1")
    monkeypatch.setenv("TRUELAYER_REDIRECT_URI", "https://app.example.com/oauth/callback")
    from app.agents.services import crypto
    from app.core.config import get_settings

    get_settings.cache_clear()
    crypto._fernet.cache_clear()
    yield
    get_settings.cache_clear()
    crypto._fernet.cache_clear()


def _patch_clients(provider: TrueLayerProvider, handler):
    transport = httpx.MockTransport(handler)

    def auth_client():
        return httpx.AsyncClient(
            base_url="https://auth.truelayer.test",
            transport=transport,
        )

    def api_client(access_token: str):
        return httpx.AsyncClient(
            base_url="https://api.truelayer.test/data/v1",
            transport=transport,
            headers={"Authorization": " ".join(("Bearer", access_token))},
        )

    return (
        patch.object(provider, "_auth_client", side_effect=auth_client),
        patch.object(provider, "_api_client", side_effect=api_client),
    )


@pytest.mark.asyncio
async def test_get_oauth_url_builds_authorization_url(truelayer_env):
    provider = TrueLayerProvider()

    url = await provider.get_oauth_url(
        "https://app.example.com/oauth/callback",
        "state-123",
        flow_params={"providers": "uk-oauth-all", "scope": "info accounts"},
    )

    assert url.startswith("https://auth.truelayer.test/?")
    query = parse_qs(url.split("?", 1)[1])
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["tl-client"]
    assert query["redirect_uri"] == ["https://app.example.com/oauth/callback"]
    assert query["scope"] == ["info accounts"]
    assert query["state"] == ["state-123"]
    assert query["providers"] == ["uk-oauth-all"]


@pytest.mark.asyncio
async def test_handle_oauth_callback_exchanges_token_and_maps_accounts(truelayer_env):
    provider = TrueLayerProvider()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/connect/token":
            body = parse_qs(request.read().decode())
            assert body["grant_type"] == ["authorization_code"]
            assert body["code"] == ["auth-code"]
            assert body["redirect_uri"] == ["https://app.example.com/oauth/callback"]
            assert body["client_id"] == ["tl-client"]
            assert body["client_secret"] == ["tl-secret"]
            return httpx.Response(
                200,
                json={
                    "access_token": "access-1",
                    "refresh_token": "refresh-1",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        if request.url.path == "/data/v1/accounts":
            assert request.headers["authorization"] == " ".join(("Bearer", "access-1"))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "account_id": "acc-1",
                            "display_name": "Current account",
                            "account_type": "TRANSACTION",
                            "currency": "GBP",
                            "account_number": {"number": "12345678"},
                            "provider": {
                                "provider_id": "lloyds",
                                "display_name": "Lloyds Bank",
                                "logo_uri": "https://logos/lloyds.png",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/data/v1/accounts/acc-1/balance":
            return httpx.Response(
                200,
                json={"results": [{"currency": "GBP", "current": "123.45"}]},
            )
        if request.url.path == "/data/v1/cards":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    auth_patch, api_patch = _patch_clients(provider, handler)
    with auth_patch, api_patch:
        conn = await provider.handle_oauth_callback("auth-code")

    assert [request.url.path for request in seen] == [
        "/connect/token",
        "/data/v1/accounts",
        "/data/v1/accounts/acc-1/balance",
        "/data/v1/cards",
    ]
    assert conn.external_id == "lloyds"
    assert conn.institution_name == "Lloyds Bank"
    assert conn.logo_url == "https://logos/lloyds.png"
    assert decrypt(conn.credentials["access_token_enc"]) == "access-1"
    assert decrypt(conn.credentials["refresh_token_enc"]) == "refresh-1"
    assert "access_token" not in conn.credentials
    assert "refresh_token" not in conn.credentials
    [account] = conn.accounts
    assert account.external_id == "acc-1"
    assert account.name == "Current account"
    assert account.type == "checking"
    assert account.balance == Decimal("123.45")
    assert account.masked_number == "5678"


@pytest.mark.asyncio
async def test_get_transactions_maps_signed_amounts_and_raw_data(truelayer_env):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/data/v1/accounts/acc-1/transactions"
        assert request.url.params["from"] == "2026-01-01"
        assert request.url.params["to"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "transaction_id": "txn-1",
                        "timestamp": "2026-01-02T12:34:56Z",
                        "description": "Coffee",
                        "merchant_name": "Cafe",
                        "amount": "-4.50",
                        "currency": "GBP",
                        "transaction_category": "Eating out",
                    },
                    {
                        "transaction_id": "txn-2",
                        "timestamp": "2026-01-03T09:00:00Z",
                        "description": "Refund",
                        "amount": "2.00",
                        "currency": "GBP",
                        "status": "pending",
                    },
                ]
            },
        )

    with _patch_clients(provider, handler)[1]:
        txns = await provider.get_transactions(
            {"access_token": "access-1"}, "acc-1", since=date(2026, 1, 1)
        )

    assert len(txns) == 2
    assert txns[0].external_id == "txn-1"
    assert txns[0].amount == Decimal("4.50")
    assert txns[0].type == "debit"
    assert txns[0].date == date(2026, 1, 2)
    assert txns[0].payee == "Cafe"
    assert txns[0].raw_data is not None
    assert txns[0].raw_data["transaction_id"] == "txn-1"
    assert txns[1].amount == Decimal("2.00")
    assert txns[1].type == "credit"
    assert txns[1].status == "pending"


@pytest.mark.asyncio
async def test_refresh_credentials_uses_refresh_token(truelayer_env):
    provider = TrueLayerProvider()
    expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    credentials = {
        "access_token": "old-access",
        "refresh_token": "refresh-1",
        "expires_at": expired,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/connect/token"
        body = parse_qs(request.read().decode())
        assert body["grant_type"] == ["refresh_token"]
        assert body["refresh_token"] == ["refresh-1"]
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "expires_in": 7200,
                "token_type": "Bearer",
            },
        )

    with _patch_clients(provider, handler)[0]:
        refreshed = await provider.refresh_credentials(credentials)

    assert _token_value(refreshed, "access_token") == "new-access"
    assert _token_value(refreshed, "refresh_token") == "refresh-1"


@pytest.mark.asyncio
async def test_valid_credentials_do_not_refresh(truelayer_env):
    provider = TrueLayerProvider()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    credentials = {"access_token": "access-1", "expires_at": future}

    refreshed = await provider.refresh_credentials(credentials)

    assert refreshed is credentials


@pytest.mark.asyncio
async def test_api_401_signals_expired_session(truelayer_env):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="expired")

    with _patch_clients(provider, handler)[1]:
        with pytest.raises(SessionExpiredError):
            await provider.get_transactions({"access_token": "bad"}, "acc-1")
