
import hashlib
import json
from dataclasses import asdict


def transaction_cost_tag(config):
    settings = asdict(config)

    serialized = json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":")
    )

    identifier = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:12]

    return f"TC{identifier}"
