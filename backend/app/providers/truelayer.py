"""TrueLayer Data API provider.

TrueLayer uses a standard OAuth authorization-code flow. Securo stores the
resulting access/refresh tokens encrypted, refreshes access tokens before sync,
and maps the Data API account/balance/transaction payloads into the shared
BankProvider dataclasses.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from app.agents.services.crypto import decrypt, encrypt
from app.core.config import get_settings
from app.providers.base import (
    AccountData,
    BankProvider,
    ConnectionData,
    ProviderRateLimited,
    SessionExpiredError,
    TransactionData,
    default_oauth_redirect_uri,
    mask_last4,
)

DEFAULT_HISTORY_DAYS = 90
TOKEN_REFRESH_BUFFER_SECONDS = 300
TRUELAYER_SCOPES = "info accounts balance transactions cards offline_access"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            parsed = _parse_datetime(value)
            return parsed.date() if parsed else None
    return None


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _account_type(raw_type: Any) -> str:
    value = str(raw_type or "").upper()
    if "SAVING" in value:
        return "savings"
    if "CARD" in value:
        return "credit_card"
    return "checking"


def _description(raw: dict) -> str:
    for key in ("description", "merchant_name", "transaction_type"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Transaction"


def _transaction_id(account_external_id: str, raw: dict) -> str:
    for key in ("transaction_id", "id", "provider_transaction_id"):
        value = raw.get(key)
        if value:
            return str(value)
    parts = [
        account_external_id,
        str(raw.get("timestamp") or raw.get("booking_date") or raw.get("date") or ""),
        str(raw.get("amount") or ""),
        str(raw.get("currency") or ""),
        _description(raw),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def _token_value(credentials: dict, key: str) -> str:
    enc = (credentials or {}).get(f"{key}_enc")
    if enc:
        decoded = decrypt(enc)
        if decoded:
            return decoded
    return (credentials or {}).get(key) or ""


def _credentials_from_token_payload(data: dict) -> dict[str, Any]:
    access_token = data.get("access_token") or ""
    refresh_token = data.get("refresh_token") or ""
    expires_in = int(data.get("expires_in") or 3600)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    credentials: dict[str, Any] = {
        "access_token_enc": encrypt(access_token) or access_token,
        "expires_at": expires_at.isoformat(),
        "scope": data.get("scope"),
        "token_type": data.get("token_type") or "Bearer",
    }
    if refresh_token:
        credentials["refresh_token_enc"] = encrypt(refresh_token) or refresh_token
    return credentials


class TrueLayerProvider(BankProvider):
    """TrueLayer open-banking connector."""

    @property
    def name(self) -> str:
        return "truelayer"

    @property
    def flow_type(self) -> str:
        return "oauth"

    @property
    def redirect_uri(self) -> str:
        return get_settings().truelayer_redirect_uri or default_oauth_redirect_uri()

    def _auth_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=get_settings().truelayer_auth_url.rstrip("/"),
            timeout=30.0,
        )

    def _api_client(self, access_token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=get_settings().truelayer_api_url.rstrip("/"),
            headers={
                "Authorization": f"******",
                "Accept": "application/json",
                "User-Agent": "Securo/0.1 (+https://usesecuro.com)",
            },
            timeout=30.0,
        )

    async def _exchange_token(self, payload: dict[str, str]) -> dict:
        settings = get_settings()
        body = {
            "client_id": settings.truelayer_client_id,
            "client_secret": settings.truelayer_client_secret.get_secret_value(),
            **payload,
        }
        async with self._auth_client() as client:
            response = await client.post("/connect/token", data=body)
        if response.status_code in (400, 401):
            raise SessionExpiredError("TrueLayer authorization expired or was rejected")
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"TrueLayer token exchange failed: {response.status_code}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def _request(
        self,
        credentials: dict,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> dict:
        access_token = _token_value(credentials, "access_token")
        if not access_token:
            raise SessionExpiredError("TrueLayer access token missing")
        async with self._api_client(access_token) as client:
            response = await client.request(method, path, params=params)
        if response.status_code in (401, 403):
            raise SessionExpiredError("TrueLayer session expired")
        if response.status_code == 429:
            raise ProviderRateLimited(f"TrueLayer {method} {path} returned 429")
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"TrueLayer {method} {path} failed: {response.status_code}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def get_oauth_url(
        self,
        redirect_uri: str,
        state: str,
        flow_params: Optional[dict] = None,
    ) -> str:
        settings = get_settings()
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": settings.truelayer_client_id,
            "redirect_uri": redirect_uri,
            "scope": str((flow_params or {}).get("scope") or TRUELAYER_SCOPES),
            "state": state,
        }
        providers = (flow_params or {}).get("providers")
        if providers:
            params["providers"] = str(providers)
        return f"{settings.truelayer_auth_url.rstrip('/')}/?{urlencode(params)}"

    async def reauth_url(
        self,
        credentials: dict,
        settings: dict,
        redirect_uri: str,
        state: str,
    ) -> str:
        return await self.get_oauth_url(
            redirect_uri,
            state,
            flow_params=(settings or {}).get("flow_params") or {},
        )

    async def handle_oauth_callback(self, code: str) -> ConnectionData:
        token_data = await self._exchange_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            }
        )
        credentials = _credentials_from_token_payload(token_data)
        accounts = await self.get_accounts(credentials)
        first = accounts[0] if accounts else None
        institution_name = first.institution_name if first and first.institution_name else "TrueLayer"
        external_id = (
            (first.institution_external_id if first else None)
            or (first.external_id if first else None)
            or hashlib.sha256(code.encode("utf-8")).hexdigest()[:32]
        )
        return ConnectionData(
            external_id=external_id,
            institution_name=institution_name,
            credentials=credentials,
            accounts=accounts,
            logo_url=first.institution_logo_url if first else None,
        )

    async def get_accounts(self, credentials: dict) -> list[AccountData]:
        data = await self._request(credentials, "GET", "/accounts")
        result: list[AccountData] = []
        for raw in data.get("results") or []:
            if isinstance(raw, dict):
                result.append(await self._build_account(credentials, raw))
        cards = await self._request(credentials, "GET", "/cards")
        for raw in cards.get("results") or []:
            if isinstance(raw, dict):
                result.append(await self._build_card(credentials, raw))
        return result

    async def _build_account(self, credentials: dict, raw: dict) -> AccountData:
        account_id = str(raw.get("account_id") or raw.get("id") or "")
        balance_raw = await self._balance(credentials, f"/accounts/{account_id}/balance")
        currency = raw.get("currency") or balance_raw.get("currency") or "GBP"
        provider = raw.get("provider") or {}
        account_number = raw.get("account_number") or {}
        return AccountData(
            external_id=account_id,
            name=raw.get("display_name") or raw.get("account_type") or "Account",
            type=_account_type(raw.get("account_type")),
            balance=_to_decimal(balance_raw.get("current") or balance_raw.get("available")) or Decimal("0"),
            currency=currency,
            masked_number=mask_last4(
                account_number.get("iban")
                or account_number.get("number")
                or account_number.get("swift_bic")
            ),
            institution_external_id=provider.get("provider_id"),
            institution_name=provider.get("display_name"),
            institution_logo_url=provider.get("logo_uri"),
        )

    async def _build_card(self, credentials: dict, raw: dict) -> AccountData:
        card_id = str(raw.get("card_id") or raw.get("id") or "")
        balance_raw = await self._balance(credentials, f"/cards/{card_id}/balance")
        provider = raw.get("provider") or {}
        return AccountData(
            external_id=f"card:{card_id}",
            name=raw.get("display_name") or "Credit card",
            type="credit_card",
            balance=(_to_decimal(balance_raw.get("current") or balance_raw.get("available")) or Decimal("0")).copy_abs(),
            currency=raw.get("currency") or balance_raw.get("currency") or "GBP",
            masked_number=mask_last4(raw.get("partial_card_number")),
            credit_limit=_to_decimal(balance_raw.get("credit_limit")),
            institution_external_id=provider.get("provider_id"),
            institution_name=provider.get("display_name"),
            institution_logo_url=provider.get("logo_uri"),
        )

    async def _balance(self, credentials: dict, path: str) -> dict:
        data = await self._request(credentials, "GET", path)
        balances = data.get("results") or []
        return balances[0] if balances and isinstance(balances[0], dict) else {}

    async def get_transactions(
        self,
        credentials: dict,
        account_external_id: str,
        since: Optional[date] = None,
        payee_source: str = "auto",
    ) -> list[TransactionData]:
        is_card = account_external_id.startswith("card:")
        resource_id = account_external_id.removeprefix("card:")
        path = f"/cards/{resource_id}/transactions" if is_card else f"/accounts/{resource_id}/transactions"
        date_from = since or (date.today() - timedelta(days=DEFAULT_HISTORY_DAYS))
        params = {"from": date_from.isoformat(), "to": date.today().isoformat()}
        data = await self._request(credentials, "GET", path, params=params)
        return [
            txn
            for raw in data.get("results") or []
            if isinstance(raw, dict)
            for txn in [self._build_transaction(account_external_id, raw, payee_source)]
            if txn is not None
        ]

    def _build_transaction(
        self,
        account_external_id: str,
        raw: dict,
        payee_source: str,
    ) -> Optional[TransactionData]:
        amount = _to_decimal(raw.get("amount"))
        if amount is None:
            return None
        txn_date = (
            _parse_date(raw.get("timestamp"))
            or _parse_date(raw.get("booking_date"))
            or _parse_date(raw.get("date"))
        )
        if txn_date is None:
            return None
        status = str(raw.get("status") or "").lower()
        return TransactionData(
            external_id=_transaction_id(account_external_id, raw),
            description=_description(raw),
            amount=amount.copy_abs(),
            date=txn_date,
            type="debit" if amount < 0 else "credit",
            currency=raw.get("currency"),
            pluggy_category=raw.get("transaction_category")
            or raw.get("provider_transaction_category"),
            status="pending" if status == "pending" else "posted",
            payee=None if payee_source in {"none", "description"} else raw.get("merchant_name"),
            raw_data=raw,
        )

    async def refresh_credentials(self, credentials: dict) -> dict:
        expires_at = _parse_datetime((credentials or {}).get("expires_at"))
        if expires_at and (
            expires_at - datetime.now(timezone.utc)
        ).total_seconds() > TOKEN_REFRESH_BUFFER_SECONDS:
            return credentials
        refresh_token = _token_value(credentials, "refresh_token")
        if not refresh_token:
            raise SessionExpiredError("TrueLayer refresh token missing")
        token_data = await self._exchange_token(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        )
        refreshed = dict(credentials or {})
        refreshed.update(_credentials_from_token_payload(token_data))
        if "refresh_token_enc" not in refreshed and refresh_token:
            refreshed["refresh_token_enc"] = encrypt(refresh_token) or refresh_token
        return refreshed
