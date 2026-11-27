"""Outbound URL and IP destination policy defending against SSRF and DNS rebinding."""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx

ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}
MAX_REDIRECTS = 5
ALLOW_PRIVATE_DOWNLOADS_ENV = "ORACLE_ALLOW_PRIVATE_DOWNLOADS"

_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes"})


class NetworkPolicyError(ValueError):
    """Raised when a download URL or resolved destination violates network policy."""


class DnsResolver(Protocol):
    """Boundary port for resolving a hostname and port into IP address strings."""

    def resolve(self, host: str, port: int) -> list[str]: ...


class SystemDnsResolver:
    """Default DNS resolver backed by ``socket.getaddrinfo``."""

    def resolve(self, host: str, port: int) -> list[str]:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise NetworkPolicyError(f"Cannot resolve host {host!r}: {exc}") from exc
        addresses: list[str] = []
        seen: set[str] = set()
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            address = str(sockaddr[0])
            if address not in seen:
                seen.add(address)
                addresses.append(address)
        if not addresses:
            raise NetworkPolicyError(f"Host {host!r} resolved to no IP addresses.")
        return addresses


@dataclass(frozen=True)
class ValidatedTarget:
    """Validated outbound HTTP(S) target and its resolved IP addresses."""

    url: str
    scheme: str
    host: str
    port: int
    resolved_ips: tuple[str, ...]
    pinned_ip: str


def _normalize_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if "%" in candidate:
        return None
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def _is_disallowed_ip_object(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_blocked_ip(address: str) -> bool:
    """Return True if ``address`` is non-global, private, loopback, link-local, or invalid."""
    if "%" in address:
        return True
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_disallowed_ip_object(ip.ipv4_mapped)
        if ip.sixtofour is not None and _is_disallowed_ip_object(ip.sixtofour):
            return True
        if ip.teredo is not None:
            server, client = ip.teredo
            if _is_disallowed_ip_object(server) or _is_disallowed_ip_object(client):
                return True
    return _is_disallowed_ip_object(ip)


class NetworkPolicy:
    """Enforce safe outbound HTTP(S) destinations, redirect bounds, and DNS rebinding checks."""

    def __init__(
        self,
        *,
        resolver: DnsResolver | None = None,
        allow_private_hosts: bool = False,
        allowed_hosts: Iterable[str] = (),
        max_redirects: int = MAX_REDIRECTS,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must be non-negative.")
        self._resolver: DnsResolver = resolver or SystemDnsResolver()
        self.allow_private_hosts = allow_private_hosts
        self.allowed_hosts: frozenset[str] = frozenset(
            _normalize_host(host) for host in allowed_hosts if host.strip()
        )
        self.max_redirects = max_redirects

    def _is_host_explicitly_allowed(self, host: str) -> bool:
        return self.allow_private_hosts or _normalize_host(host) in self.allowed_hosts

    def _resolve_and_check_host(self, host: str, port: int) -> tuple[str, ...]:
        host_allowed = self._is_host_explicitly_allowed(host)
        ip_literal = _parse_ip_literal(host)
        if ip_literal is not None:
            resolved = [str(ip_literal)]
        elif "%" in host:
            raise NetworkPolicyError(f"Blocked scoped IP literal for host {host!r}.")
        else:
            resolved = self._resolver.resolve(host, port)
        if not resolved:
            raise NetworkPolicyError(f"Host {host!r} resolved to no IP addresses.")
        deduped: list[str] = []
        seen: set[str] = set()
        for address in resolved:
            if address in seen:
                continue
            seen.add(address)
            if not host_allowed and is_blocked_ip(address):
                raise NetworkPolicyError(
                    f"Blocked non-public destination address {address!r} for host {host!r}."
                )
            deduped.append(address)
        return tuple(deduped)

    def validate_url(self, raw_url: str) -> ValidatedTarget:
        """Validate URL syntax, scheme, absence of credentials, and resolved IP addresses."""
        try:
            url = httpx.URL(raw_url)
        except httpx.InvalidURL as exc:
            raise NetworkPolicyError(f"Invalid download URL: {exc}") from exc
        if url.scheme not in ALLOWED_SCHEMES:
            raise NetworkPolicyError(
                f"Disallowed URL scheme {url.scheme!r}; only http and https are permitted."
            )
        parsed = urlsplit(raw_url)
        if url.username or url.password or "@" in parsed.netloc:
            raise NetworkPolicyError("Download URLs must not contain embedded credentials.")
        host = url.host
        if not host:
            raise NetworkPolicyError("Download URL must include a non-empty host.")
        port = url.port or DEFAULT_PORTS[url.scheme]
        resolved_ips = self._resolve_and_check_host(host, port)
        return ValidatedTarget(
            url=str(url),
            scheme=url.scheme,
            host=host,
            port=port,
            resolved_ips=resolved_ips,
            pinned_ip=resolved_ips[0],
        )

    def verify_before_connect(self, target: ValidatedTarget) -> ValidatedTarget:
        """Re-check DNS immediately before connecting to detect DNS rebinding."""
        if _parse_ip_literal(target.host) is not None:
            return target
        refreshed = self.validate_url(target.url)
        if not self._is_host_explicitly_allowed(target.host):
            if not (set(refreshed.resolved_ips) & set(target.resolved_ips)):
                raise NetworkPolicyError(
                    f"DNS rebinding detected for host {target.host!r}: resolved IPs changed."
                )
        return refreshed

    def pinned_request_target(
        self, target: ValidatedTarget, *, ip: str | None = None
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        """Build a pinned-IP URL, Host header, and TLS SNI extension for direct connections."""
        chosen_ip = target.pinned_ip if ip is None else ip
        if chosen_ip not in target.resolved_ips:
            raise NetworkPolicyError(
                f"Requested IP {chosen_ip!r} is not in validated addresses for {target.host!r}."
            )
        parsed = httpx.URL(target.url)
        pinned_host = f"[{chosen_ip}]" if ":" in chosen_ip else chosen_ip
        pinned_url = str(parsed.copy_with(host=pinned_host, port=target.port))
        default_port = DEFAULT_PORTS[target.scheme]
        host_header = (
            target.host if target.port == default_port else f"{target.host}:{target.port}"
        )
        headers = {"Host": host_header}
        extensions: dict[str, object] = {}
        if target.scheme == "https" and _parse_ip_literal(target.host) is None:
            extensions["sni_hostname"] = target.host
        return pinned_url, headers, extensions


def default_network_policy() -> NetworkPolicy:
    """Build the default download policy, honoring ``ORACLE_ALLOW_PRIVATE_DOWNLOADS`` opt-in."""
    allow_private = (
        os.environ.get(ALLOW_PRIVATE_DOWNLOADS_ENV, "").strip().casefold() in _TRUTHY_ENV_VALUES
    )
    return NetworkPolicy(allow_private_hosts=allow_private)
