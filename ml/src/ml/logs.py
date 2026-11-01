"""Synthetic service logs: the kind of text Sentinel will read all day.

A small preview of the Arena (Module 14): 5 services, access logs, JSON warnings, errors.
"""

import json
import random
import uuid

SERVICES = ["api-gateway", "payments", "orders", "auth", "inventory"]
PATHS = ["/api/v1/users", "/api/v1/orders", "/api/v1/payments/charge", "/health", "/api/v1/inventory/items"]
ERRORS = [
    "connection refused",
    "timeout after {ms}ms",
    "deadlock detected",
    "pool exhausted (max=20)",
    "invalid token",
    "upstream returned 503",
]


def generate_logs(n: int, seed: int = 42) -> list[str]:
    """n log lines. Same seed → same lines, every time."""
    rng = random.Random(seed)  # own generator: doesn't disturb anyone else's random.seed()

    def ts(i: int) -> str:
        return f"2026-10-19T{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}.{rng.randint(0, 999):03d}Z"

    def line(i: int) -> str:
        kind = rng.random()
        svc = rng.choice(SERVICES)
        rid = str(uuid.UUID(int=rng.getrandbits(128)))
        if kind < 0.5:  # 50%: access log
            return (
                f"{ts(i)} INFO  {svc} request_id={rid} method=GET "
                f"path={rng.choice(PATHS)} status=200 duration_ms={rng.randint(3, 250)}"
            )
        if kind < 0.75:  # 25%: structured JSON warning
            return json.dumps(
                {
                    "ts": ts(i),
                    "level": "warn",
                    "service": svc,
                    "msg": "slow query",
                    "duration_ms": rng.randint(500, 5000),
                    "trace_id": rid.replace("-", ""),
                }
            )
        # 25%: error
        err = rng.choice(ERRORS).format(ms=rng.randint(1000, 30000))
        return f'{ts(i)} ERROR {svc} request_id={rid} msg="{err}" retry={rng.randint(0, 3)}'

    return [line(i) for i in range(n)]
