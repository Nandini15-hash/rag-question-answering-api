"""
Rate limiting via slowapi (a thin wrapper around the `limits` library).

Chosen over hand-rolling a token bucket because slowapi is ~200 lines,
integrates with FastAPI's dependency/exception-handler system directly, and
already handles the edge cases (per-key buckets, proper 429 responses,
Retry-After headers) that a hand-rolled version would need to reinvent —
"avoid heavy frameworks unless justified" cuts against something like
Django-only solutions, not a small purpose-built library like this one.
Limits are per-client-IP and in-memory, which is enough for a single-process
assignment deployment (a multi-worker/multi-node deployment would swap the
in-memory backend for Redis, a one-line change in slowapi).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
