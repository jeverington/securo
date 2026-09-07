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
        "/data/v1/cards",
        "/data/v1/accounts/acc-1/balance",
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


@pytest.mark.asyncio
async def test_cards_are_keyed_by_account_id(truelayer_env):
    """TrueLayer's Cards API has no `card_id` — cards carry `account_id`."""
    provider = TrueLayerProvider()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/cards":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "account_id": "card-acc-1",
                            "display_name": "Visa Credit",
                            "card_type": "CREDIT",
                            "card_network": "VISA",
                            "currency": "GBP",
                            "partial_card_number": "4444",
                            "provider": {
                                "provider_id": "amex",
                                "display_name": "Amex",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/data/v1/cards/card-acc-1/balance":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "currency": "GBP",
                            "current": "250.00",
                            "credit_limit": "3000.00",
                            "payment_due": "25.00",
                            "payment_due_date": "2026-02-17T00:00:00Z",
                            "last_statement_balance": "250.00",
                            "last_statement_date": "2026-01-24T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(404, text="not found")

    with _patch_clients(provider, handler)[1]:
        accounts = await provider.get_accounts({"access_token": "access-1"})

    assert "/data/v1/cards/card-acc-1/balance" in seen
    [card] = accounts
    assert card.external_id == "card:card-acc-1"
    assert card.type == "credit_card"
    assert card.balance == Decimal("250.00")
    assert card.credit_limit == Decimal("3000.00")
    assert card.masked_number == "4444"
    assert card.minimum_payment == Decimal("25.00")
    assert card.payment_due_day == 17
    assert card.statement_close_day == 24
    assert card.card_brand == "VISA"


@pytest.mark.asyncio
async def test_card_transactions_use_the_card_endpoint(truelayer_env):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/data/v1/cards/card-acc-1/transactions"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "transaction_id": "txn-1",
                        "timestamp": "2026-01-02T12:34:56Z",
                        "description": "Hotel",
                        "amount": "-99.00",
                        "currency": "GBP",
                    }
                ]
            },
        )

    with _patch_clients(provider, handler)[1]:
        txns = await provider.get_transactions(
            {"access_token": "access-1"}, "card:card-acc-1"
        )

    assert [txn.external_id for txn in txns] == ["txn-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 501])
async def test_providers_without_card_support_still_connect(truelayer_env, status_code):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "account_id": "acc-1",
                            "display_name": "Current account",
                            "account_type": "TRANSACTION",
                            "currency": "GBP",
                        }
                    ]
                },
            )
        if request.url.path == "/data/v1/accounts/acc-1/balance":
            return httpx.Response(
                200, json={"results": [{"currency": "GBP", "current": "10.00"}]}
            )
        if request.url.path == "/data/v1/cards":
            return httpx.Response(status_code, text="endpoint_not_supported")
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        accounts = await provider.get_accounts({"access_token": "access-1"})

    assert [account.external_id for account in accounts] == ["acc-1"]


@pytest.mark.asyncio
async def test_api_error_message_includes_provider_body(truelayer_env):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='{"error":"internal_server_error"}')

    with _patch_clients(provider, handler)[1]:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await provider.get_accounts({"access_token": "access-1"})

    assert "internal_server_error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_zero_current_balance_is_not_replaced_by_available(truelayer_env):
    """A paid-off card must read 0, not its whole credit limit."""
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/cards":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"account_id": "card-1", "display_name": "Visa", "currency": "GBP"}
                    ]
                },
            )
        if request.url.path == "/data/v1/cards/card-1/balance":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "currency": "GBP",
                            "current": 0,
                            "available": 3300,
                            "credit_limit": 3300,
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [card] = await provider.get_accounts({"access_token": "access-1"})

    assert card.balance == Decimal("0")
    assert card.credit_limit == Decimal("3300")


@pytest.mark.asyncio
async def test_unreadable_balance_leaves_the_account_at_zero(truelayer_env):
    """A balance the bank will never serve must not fail the connection.

    Amex refuses supplementary-card balances for the life of the connection.
    """
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "account_id": "acc-1",
                            "display_name": "Current account",
                            "account_type": "TRANSACTION",
                            "currency": "GBP",
                        }
                    ]
                },
            )
        if request.url.path == "/data/v1/accounts/acc-1/balance":
            return httpx.Response(404, text="account_not_found")
        if request.url.path == "/data/v1/cards":
            return httpx.Response(200, json={"results": []})
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [account] = await provider.get_accounts({"access_token": "access-1"})

    assert account.external_id == "acc-1"
    assert account.balance == Decimal("0")
    assert account.currency == "GBP"


@pytest.mark.asyncio
async def test_card_metadata_is_omitted_when_the_provider_has_none(truelayer_env):
    """Partial payloads must leave the cycle fields unset, not zeroed.

    connection_service only overwrites stored CC metadata when the provider
    supplies a value, so None here preserves a user's manual entry.
    """
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/cards":
            return httpx.Response(
                200,
                json={"results": [{"account_id": "card-1", "display_name": "Card"}]},
            )
        if request.url.path == "/data/v1/cards/card-1/balance":
            return httpx.Response(
                200, json={"results": [{"currency": "GBP", "current": "10.00"}]}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [card] = await provider.get_accounts({"access_token": "access-1"})

    assert card.credit_limit is None
    assert card.minimum_payment is None
    assert card.payment_due_day is None
    assert card.statement_close_day is None
    assert card.card_brand is None
    assert card.card_level is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 501])
async def test_card_only_issuers_connect_without_an_accounts_endpoint(
    truelayer_env, status_code
):
    """Barclaycard/Amex expose no /accounts at all."""
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(status_code, text="endpoint_not_supported")
        if request.url.path == "/data/v1/cards":
            return httpx.Response(
                200,
                json={"results": [{"account_id": "card-1", "display_name": "Amex"}]},
            )
        if request.url.path == "/data/v1/cards/card-1/balance":
            return httpx.Response(
                200, json={"results": [{"currency": "GBP", "current": "40.00"}]}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [card] = await provider.get_accounts({"access_token": "access-1"})

    assert card.external_id == "card:card-1"
    assert card.balance == Decimal("40.00")


@pytest.mark.asyncio
async def test_neither_endpoint_available_is_an_error_not_an_empty_bank(truelayer_env):
    """A wrong API base 404s both lists; that must not read as "no accounts"."""
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="endpoint_not_supported")

    with _patch_clients(provider, handler)[1]:
        with pytest.raises(RuntimeError, match="neither /accounts nor /cards"):
            await provider.get_accounts({"access_token": "access-1"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, text="upstream_error"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["server_error", "timeout"],
)
async def test_transient_balance_failures_propagate(truelayer_env, failure):
    """Sync overwrites account.balance unconditionally.

    A transient failure must abort rather than persist a zero over a real
    balance, so only permanently-unreadable balances degrade.
    """
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "account_id": "acc-1",
                            "display_name": "Current account",
                            "currency": "GBP",
                        }
                    ]
                },
            )
        if request.url.path == "/data/v1/cards":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/accounts/acc-1/balance":
            if isinstance(failure, httpx.Response):
                return failure
            raise failure
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        with pytest.raises(httpx.HTTPError):
            await provider.get_accounts({"access_token": "access-1"})


@pytest.mark.asyncio
async def test_expired_session_during_balance_still_triggers_reauth(truelayer_env):
    """401 is never tolerated — the connection must be flagged for reconnect."""
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(
                200,
                json={"results": [{"account_id": "acc-1", "display_name": "Account"}]},
            )
        if request.url.path == "/data/v1/cards":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(401, text="invalid_token")

    with _patch_clients(provider, handler)[1]:
        with pytest.raises(SessionExpiredError):
            await provider.get_accounts({"access_token": "access-1"})


@pytest.mark.asyncio
async def test_card_balance_never_falls_back_to_available_credit(truelayer_env):
    """`available` on a card is headroom, the inverse of what it owes."""
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/cards":
            return httpx.Response(
                200,
                json={"results": [{"account_id": "card-1", "display_name": "Card"}]},
            )
        if request.url.path == "/data/v1/cards/card-1/balance":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"currency": "GBP", "available": "3279.00", "credit_limit": "3300.00"}
                    ]
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [card] = await provider.get_accounts({"access_token": "access-1"})

    assert card.balance == Decimal("0")
    assert card.credit_limit == Decimal("3300.00")


@pytest.mark.asyncio
async def test_current_account_still_falls_back_to_available(truelayer_env):
    provider = TrueLayerProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data/v1/accounts":
            return httpx.Response(
                200,
                json={"results": [{"account_id": "acc-1", "display_name": "Account"}]},
            )
        if request.url.path == "/data/v1/cards":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/data/v1/accounts/acc-1/balance":
            return httpx.Response(
                200, json={"results": [{"currency": "GBP", "available": "12.50"}]}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    with _patch_clients(provider, handler)[1]:
        [account] = await provider.get_accounts({"access_token": "access-1"})

    assert account.balance == Decimal("12.50")
