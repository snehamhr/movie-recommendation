from __future__ import annotations

import os

from recommender import (
    POSTER_METADATA_PATH,
    build_full_poster_manifest,
    ensure_recommender,
)


def main() -> None:
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Set TMDB_API_KEY before running this script.\n"
            'PowerShell: $env:TMDB_API_KEY="your_v3_api_key"'
        )

    engine = ensure_recommender()

    manifest = build_full_poster_manifest(
        engine,
        api_key,
        max_workers=6,
        batch_size=100,
    )

    print(f"Saved: {POSTER_METADATA_PATH}")
    print(f"Movies cached: {len(manifest):,}")


if __name__ == "__main__":
    main()