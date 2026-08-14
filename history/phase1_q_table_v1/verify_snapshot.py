"""Verify the frozen phase-one payload and optionally re-evaluate its policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_payload() -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_json(ARCHIVE_ROOT / "manifest.json")
    expected_hashes = manifest["sha256"]
    if not isinstance(expected_hashes, dict):
        raise TypeError("manifest sha256 field must be an object")

    failures: list[str] = []
    for relative_path, expected_hash in expected_hashes.items():
        payload_path = ARCHIVE_ROOT / relative_path
        if not payload_path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            failures.append(f"changed: {relative_path}")

    if failures:
        raise RuntimeError("snapshot verification failed:\n" + "\n".join(failures))

    checkpoint = load_json(ARCHIVE_ROOT / "checkpoints" / "q_table.json")
    metrics = load_json(ARCHIVE_ROOT / "runs" / "q_learning.json")
    q_values = checkpoint.get("q_values")
    if not isinstance(q_values, dict) or len(q_values) != manifest["q_states"]:
        raise RuntimeError("checkpoint Q-state count does not match manifest")
    if checkpoint.get("state_encoding") != manifest["state_encoding"]:
        raise RuntimeError("checkpoint state encoding does not match manifest")
    saved = dict(manifest["saved_evaluation"])
    saved.pop("episodes")
    if metrics.get("evaluation") != saved:
        raise RuntimeError("saved evaluation does not match manifest")

    return checkpoint, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="repeat the fixed-seed 100-episode policy evaluation",
    )
    args = parser.parse_args()

    checkpoint, manifest = verify_payload()
    print(
        f"snapshot={manifest['archive_id']} status=verified "
        f"files={len(manifest['sha256'])} q_states={manifest['q_states']}"
    )

    if args.evaluate:
        from hk_rl.train_q import evaluate

        expected = manifest["verification_evaluation"]
        result = evaluate(
            checkpoint,
            episodes=expected["episodes"],
            seed=expected["seed"],
        )
        print(json.dumps(result, indent=2))
        for key in (
            "win_rate",
            "average_damage_taken",
            "average_spike_escape_timeouts",
        ):
            if result[key] != expected[key]:
                raise RuntimeError(f"evaluation mismatch for {key}")


if __name__ == "__main__":
    main()
