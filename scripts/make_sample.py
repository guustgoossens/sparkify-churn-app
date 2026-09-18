"""Build the small, reproducible event sample that ships with the repository.

The full competition file (``train.parquet``, ~17.5M events, ~600 MB) is far too
large to version. This script draws a stratified random sample of users (same
churn rate as the full data), keeps *all* events for those users, and drops the
columns the app never uses (names, user agent, ...).

Usage:
    uv run python scripts/make_sample.py /path/to/train.parquet
    uv run python scripts/make_sample.py /path/to/train.parquet --users 500 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHURN_PAGE = "Cancellation Confirmation"
KEPT_COLUMNS = [
    "userId",
    "sessionId",
    "itemInSession",
    "time",
    "registration",
    "page",
    "level",
    "gender",
    "location",
    "status",
    "length",
    "song",
    "artist",
]


def sample_user_ids(source: Path, n_users: int, seed: int) -> list[str]:
    """Pick ``n_users`` user ids, preserving the churn rate of the full data."""
    slim = pq.read_table(source, columns=["userId", "page"])
    all_users = pc.unique(slim["userId"]).to_numpy(zero_copy_only=False)
    churned = pc.unique(
        slim.filter(pc.equal(slim["page"], CHURN_PAGE))["userId"]
    ).to_numpy(zero_copy_only=False)
    retained = np.setdiff1d(all_users, churned)

    rng = np.random.default_rng(seed)
    n_churned = round(n_users * len(churned) / len(all_users))
    picked = np.concatenate(
        [
            rng.choice(np.sort(churned), size=n_churned, replace=False),
            rng.choice(np.sort(retained), size=n_users - n_churned, replace=False),
        ]
    )
    return sorted(picked.tolist())


def build_sample(source: Path, destination: Path, n_users: int, seed: int) -> pa.Table:
    user_ids = sample_user_ids(source, n_users, seed)
    table = pq.read_table(
        source,
        columns=KEPT_COLUMNS,
        filters=[("userId", "in", user_ids)],
    )
    table = table.sort_by([("userId", "ascending"), ("time", "ascending")])
    # Strip the pandas index metadata so the file is identical across reruns.
    table = table.replace_schema_metadata(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd", compression_level=19)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="Path to the full train.parquet")
    parser.add_argument(
        "--output", type=Path, default=Path("data/sample_events.parquet")
    )
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = build_sample(args.source, args.output, args.users, args.seed)
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"Wrote {table.num_rows:,} events for {args.users} users "
        f"to {args.output} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
