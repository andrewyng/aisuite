"""OAuth 2.0 (authorization-code + PKCE) for remote MCP servers.

Built for Google's hosted Workspace MCP servers (Calendar/Gmail) but provider-agnostic:
any authorization server speaking the standard authorization-code + refresh-token grants
works via `authorize_url`/`token_url` overrides. The interactive consent runs once in the
system browser with a localhost loopback redirect; tokens are cached in the SecretStore
(profile `mcp-oauth:<server>`) so later sessions refresh silently. Google does NOT support
dynamic client registration, so the OAuth client id/secret come from config.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets as pysecrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

import httpx

from ..secrets import SecretStore

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_CALLBACK_PORT = 51789
CALLBACK_PATH = "/oauth/callback"
_EXPIRY_SLACK = 60.0  # treat tokens expiring within this window as already expired
_FLOW_TIMEOUT = 300.0  # seconds the user has to finish the browser consent

_SUCCESS_HTML = (
    "<html><body style='font-family:sans-serif;text-align:center;padding-top:4em'>"
    "<h2>Connected</h2><p>Authorization complete &mdash; you can close this tab and "
    "return to OpenCoworker.</p></body></html>"
)
_FAILURE_HTML = (
    "<html><body style='font-family:sans-serif;text-align:center;padding-top:4em'>"
    "<h2>Authorization failed</h2><p>{reason}</p></body></html>"
)


class OAuthError(RuntimeError):
    pass


@dataclass
class OAuthSpec:
    """Static OAuth client settings for one MCP server (defaults target Google)."""

    client_id: str
    client_secret: str = ""
    scopes: list[str] = field(default_factory=list)
    authorize_url: str = GOOGLE_AUTHORIZE_URL
    token_url: str = GOOGLE_TOKEN_URL
    callback_port: int = DEFAULT_CALLBACK_PORT

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.callback_port}{CALLBACK_PATH}"

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "OAuthSpec":
        client_id = str(raw.get("client_id") or "")
        if not client_id:
            raise ValueError("MCP oauth config requires a client_id")
        return cls(
            client_id=client_id,
            client_secret=str(raw.get("client_secret") or ""),
            scopes=[str(s) for s in (raw.get("scopes") or [])],
            authorize_url=str(raw.get("authorize_url") or GOOGLE_AUTHORIZE_URL),
            token_url=str(raw.get("token_url") or GOOGLE_TOKEN_URL),
            callback_port=int(raw.get("callback_port") or DEFAULT_CALLBACK_PORT),
        )


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(pysecrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_authorize_url(spec: OAuthSpec, state: str, challenge: str) -> str:
    params = {
        "client_id": spec.client_id,
        "redirect_uri": spec.redirect_uri,
        "response_type": "code",
        "scope": " ".join(spec.scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # Google-specific but harmless elsewhere: offline => refresh_token issued;
        # prompt=consent forces re-issue even if previously granted.
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{spec.authorize_url}?{urllib.parse.urlencode(params)}"


async def _start_callback_server(
    port: int, expected_state: str
) -> tuple[asyncio.AbstractServer, "asyncio.Future[str]"]:
    """Bind the loopback redirect server; the future resolves to the auth code."""
    loop = asyncio.get_running_loop()
    result: asyncio.Future[str] = loop.create_future()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        status, body = "200 OK", _SUCCESS_HTML
        try:
            request_line = await reader.readline()
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass  # drain headers
            parts = request_line.split()
            target = parts[1].decode("latin-1") if len(parts) >= 2 else "/"
            parsed = urllib.parse.urlsplit(target)
            if parsed.path != CALLBACK_PATH:
                status, body = "404 Not Found", _FAILURE_HTML.format(reason="Not found")
            else:
                query = urllib.parse.parse_qs(parsed.query)
                error = (query.get("error") or [None])[0]
                code = (query.get("code") or [None])[0]
                state = (query.get("state") or [None])[0]
                if error or not code:
                    reason = error or "no authorization code in callback"
                    status = "400 Bad Request"
                    body = _FAILURE_HTML.format(reason=reason)
                    if not result.done():
                        result.set_exception(
                            OAuthError(f"authorization failed: {reason}")
                        )
                elif state != expected_state:
                    status = "400 Bad Request"
                    body = _FAILURE_HTML.format(reason="state mismatch")
                    if not result.done():
                        result.set_exception(OAuthError("state mismatch in callback"))
                elif not result.done():
                    result.set_result(code)
        finally:
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
                f"Content-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}".encode()
            )
            try:
                await writer.drain()
            except OSError:
                pass
            writer.close()

    server = await asyncio.start_server(handle, host="localhost", port=port)
    return server, result


class OAuthTokenManager:
    """Caches/refreshes one MCP server's tokens; runs browser consent when needed."""

    def __init__(
        self,
        server_name: str,
        spec: OAuthSpec,
        secrets: Optional[SecretStore] = None,
        *,
        open_browser: Callable[[str], Any] = webbrowser.open,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        flow_timeout: float = _FLOW_TIMEOUT,
    ) -> None:
        self.spec = spec
        self._profile = f"mcp-oauth:{server_name}"
        self._secrets = secrets or SecretStore()
        self._open_browser = open_browser
        self._transport = transport
        self._flow_timeout = flow_timeout
        self._lock = asyncio.Lock()

    async def get_token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            tokens = self._secrets.get(self._profile) or {}
            fresh = tokens.get("expires", 0) > time.time() + _EXPIRY_SLACK
            if tokens.get("access_token") and fresh and not force_refresh:
                return str(tokens["access_token"])
            if tokens.get("refresh_token"):
                try:
                    tokens = await self._refresh(tokens)
                except OAuthError:
                    tokens = await self._interactive()
            else:
                tokens = await self._interactive()
            return str(tokens["access_token"])

    # -- grant flows --------------------------------------------------------------
    async def _refresh(self, tokens: dict[str, Any]) -> dict[str, Any]:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": self.spec.client_id,
        }
        if self.spec.client_secret:
            data["client_secret"] = self.spec.client_secret
        payload = await self._post_token(data)
        return self._store(payload, previous=tokens)

    async def _interactive(self) -> dict[str, Any]:
        verifier, challenge = _pkce_pair()
        state = pysecrets.token_urlsafe(16)
        url = build_authorize_url(self.spec, state, challenge)
        server, result = await _start_callback_server(self.spec.callback_port, state)
        try:
            self._open_browser(url)
            code = await asyncio.wait_for(result, self._flow_timeout)
        except asyncio.TimeoutError:
            raise OAuthError(
                f"timed out after {int(self._flow_timeout)}s waiting for "
                "browser authorization"
            ) from None
        finally:
            server.close()
            await server.wait_closed()
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.spec.redirect_uri,
            "client_id": self.spec.client_id,
            "code_verifier": verifier,
        }
        if self.spec.client_secret:
            data["client_secret"] = self.spec.client_secret
        payload = await self._post_token(data)
        return self._store(payload, previous={})

    # -- plumbing -------------------------------------------------------------------
    async def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        async with httpx.AsyncClient(transport=self._transport, timeout=30) as client:
            response = await client.post(self.spec.token_url, data=data)
        if response.status_code != 200:
            raise OAuthError(
                f"token endpoint returned {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    def _store(
        self, payload: dict[str, Any], *, previous: dict[str, Any]
    ) -> dict[str, Any]:
        access_token = payload.get("access_token")
        if not access_token:
            raise OAuthError("token endpoint response had no access_token")
        tokens = {
            "type": "oauth",
            "access_token": access_token,
            # Google omits refresh_token on refresh responses — keep the old one.
            "refresh_token": payload.get("refresh_token")
            or previous.get("refresh_token"),
            "expires": time.time() + float(payload.get("expires_in") or 3600),
            "scopes": self.spec.scopes,
        }
        self._secrets.put(self._profile, tokens)
        return tokens


class OAuthBearer(httpx.Auth):
    """httpx auth hook: fresh Bearer token per request, one forced refresh on 401."""

    def __init__(self, manager: OAuthTokenManager) -> None:
        self._manager = manager

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {await self._manager.get_token()}"
        response = yield request
        if response is not None and response.status_code == 401:
            token = await self._manager.get_token(force_refresh=True)
            request.headers["Authorization"] = f"Bearer {token}"
            yield request


def build_auth(server_name: str, oauth_config: dict[str, Any]) -> OAuthBearer:
    """Config dict -> ready `httpx.Auth` for `streamablehttp_client`."""
    spec = OAuthSpec.from_config(oauth_config)
    return OAuthBearer(OAuthTokenManager(server_name, spec))
