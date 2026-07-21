from __future__ import annotations

import ast
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from nltk.stem.porter import PorterStemmer
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib3.util.retry import Retry


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "movie_recommender.joblib"
POSTER_METADATA_PATH = ARTIFACT_DIR / "poster_metadata.json"

KAGGLE_DATASET_HANDLE = "tmdb/tmdb-movie-metadata"
MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"

PLACEHOLDER_BASE = "https://placehold.co/500x750/151827/F8FAFC.png"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_MANIFEST_LOCK = threading.Lock()

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "CineMatch/1.0",
        "Accept": "application/json",
    }
)
_SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
    ),
)


class RecommenderError(RuntimeError):
    """Raised when the recommendation engine cannot be prepared or loaded."""


def _find_file(directory: Path, filename: str) -> Path | None:
    if not directory.exists():
        return None

    direct = directory / filename
    if direct.is_file():
        return direct

    matches = list(directory.rglob(filename))
    return matches[0] if matches else None


def resolve_dataset_files() -> tuple[Path, Path]:
    search_dirs = [
        DATA_DIR,
        PROJECT_DIR,
        Path.cwd(),
        Path("/kaggle/input/tmdb-movie-metadata"),
        Path("/kaggle/input/datasets/tmdb/tmdb-movie-metadata"),
    ]

    for directory in search_dirs:
        movies = _find_file(directory, MOVIES_FILENAME)
        credits = _find_file(directory, CREDITS_FILENAME)

        if movies and credits:
            return movies, credits

    try:
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
        movies = _find_file(downloaded, MOVIES_FILENAME)
        credits = _find_file(downloaded, CREDITS_FILENAME)

        if movies and credits:
            return movies, credits

    except Exception as exc:
        raise RecommenderError(
            "TMDB dataset could not be found or downloaded. Place "
            f"`{MOVIES_FILENAME}` and `{CREDITS_FILENAME}` inside the `data` folder. "
            f"Download error: {exc}"
        ) from exc

    raise RecommenderError(
        f"Could not locate `{MOVIES_FILENAME}` and `{CREDITS_FILENAME}`."
    )


def convert(obj: str) -> list[str]:
    """Extract every name from a TMDB JSON-like list."""

    return [item["name"] for item in ast.literal_eval(obj)]


def convert_cast(obj: str) -> list[str]:
    """Extract the first three cast members."""

    output: list[str] = []

    for item in ast.literal_eval(obj):
        if len(output) == 3:
            break
        output.append(item["name"])

    return output


def fetch_director(obj: str) -> list[str]:
    """Extract the first crew member whose job is Director."""

    for item in ast.literal_eval(obj):
        if item.get("job") == "Director":
            return [item["name"]]

    return []


def _remove_spaces(values: list[str]) -> list[str]:
    return [value.replace(" ", "") for value in values]


_STEMMER = PorterStemmer()


def convert_stem(text: str) -> str:
    """Apply Porter stemming word by word."""

    return " ".join(_STEMMER.stem(word) for word in text.split())


def build_recommender() -> dict[str, Any]:
    movies_path, credits_path = resolve_dataset_files()

    credits = pd.read_csv(credits_path)
    movies = pd.read_csv(movies_path)

    movies = movies.merge(credits, on="title")
    movies = movies[
        ["title", "genres", "id", "overview", "keywords", "cast", "crew"]
    ].copy()
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda value: value.split())

    for column in ["genres", "overview", "keywords", "cast", "crew"]:
        movies[column] = movies[column].apply(_remove_spaces)

    movies["tags"] = (
        movies["genres"]
        + movies["overview"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    frame = movies[["id", "title", "tags"]].copy()
    frame["tags"] = frame["tags"].apply(lambda values: " ".join(values).lower())
    frame["tags"] = frame["tags"].apply(convert_stem)
    frame.drop_duplicates(subset=["title"], keep="first", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
    )
    vectors = vectorizer.fit_transform(frame["tags"])

    bundle = {
        "movies": frame[["id", "title"]].copy(),
        "tags": frame["tags"].copy(),
        "vectors": vectors,
        "vectorizer": vectorizer,
        "method": (
            "CountVectorizer(max_features=5000, stop_words='english') "
            "+ cosine similarity"
        ),
        "dataset_movies_file": movies_path.name,
        "dataset_credits_file": credits_path.name,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, ARTIFACT_PATH)

    return bundle


def _validate_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise RecommenderError("Saved recommender artifact is invalid.")

    required = {"movies", "vectors", "vectorizer"}
    missing = required - set(bundle)

    if missing:
        raise RecommenderError(
            "Saved recommender artifact is missing: " + ", ".join(sorted(missing))
        )

    return bundle


def ensure_recommender() -> dict[str, Any]:
    if ARTIFACT_PATH.is_file():
        try:
            return _validate_bundle(joblib.load(ARTIFACT_PATH))
        except Exception:
            ARTIFACT_PATH.unlink(missing_ok=True)

    return build_recommender()


def recommend_movies(
    movie_title: str,
    engine: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    movies = engine["movies"]
    matches = movies.index[movies["title"].eq(movie_title)].tolist()

    if not matches:
        raise RecommenderError(f"Movie not found: {movie_title}")

    movie_index = matches[0]
    selected_vector = engine["vectors"][movie_index]
    similarities = cosine_similarity(selected_vector, engine["vectors"]).ravel()

    ranked_indices = np.argsort(similarities)[::-1]
    ranked_indices = [
        index for index in ranked_indices if index != movie_index
    ][:top_n]

    return [
        {
            "id": int(movies.iloc[index]["id"]),
            "title": str(movies.iloc[index]["title"]),
            "similarity": float(similarities[index]),
        }
        for index in ranked_indices
    ]


def _placeholder_url(title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9 ]+", "", title).strip().replace(" ", "+")
    return f"{PLACEHOLDER_BASE}?text={safe or 'Movie'}"


def _manifest_key(movie_id: int) -> str:
    return str(int(movie_id))


@lru_cache(maxsize=1)
def load_poster_manifest() -> dict[str, dict[str, Any]]:
    """Load the committed poster metadata cache once per process."""

    if not POSTER_METADATA_PATH.is_file():
        return {}

    try:
        payload = json.loads(POSTER_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def save_poster_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    """Atomically save poster metadata.

    On Streamlit Community Cloud this runtime write may be temporary. For fast
    deployments, generate this file locally and commit it to GitHub.
    """

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with _MANIFEST_LOCK:
        temporary = POSTER_METADATA_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, POSTER_METADATA_PATH)


@lru_cache(maxsize=8192)
def _tmdb_details(movie_id: int, api_key: str) -> dict[str, Any]:
    response = _SESSION.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={
            "api_key": api_key,
            "language": "en-US",
        },
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=8192)
def _tmdb_search(title: str, api_key: str) -> dict[str, Any]:
    response = _SESSION.get(
        "https://api.themoviedb.org/3/search/movie",
        params={
            "api_key": api_key,
            "query": title,
            "include_adult": "false",
            "language": "en-US",
        },
        timeout=12,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    return results[0] if results else {}


def _normalise_visual(
    movie_id: int,
    title: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    poster_path = data.get("poster_path")
    release_date = str(data.get("release_date") or "")
    rating = data.get("vote_average")

    return {
        "id": int(movie_id),
        "title": title,
        "poster_url": (
            f"{TMDB_IMAGE_BASE}{poster_path}"
            if poster_path
            else _placeholder_url(title)
        ),
        "year": release_date[:4] if len(release_date) >= 4 else None,
        "rating": float(rating) if rating is not None else None,
        "overview": data.get("overview"),
    }


def _fetch_visual_from_tmdb(
    movie_id: int,
    title: str,
    api_key: str | None,
) -> dict[str, Any]:
    if not api_key:
        return _normalise_visual(movie_id, title, {})

    direct_data: dict[str, Any] = {}

    try:
        direct_data = _tmdb_details(movie_id, api_key)
        if direct_data.get("poster_path"):
            return _normalise_visual(movie_id, title, direct_data)
    except Exception:
        pass

    try:
        search_data = _tmdb_search(title, api_key)
        if search_data:
            return _normalise_visual(movie_id, title, search_data)
    except Exception:
        pass

    return _normalise_visual(movie_id, title, direct_data)


def get_movie_visual(
    movie_id: int,
    title: str,
    api_key: str | None,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Return poster/details from the prebuilt manifest or fetch once from TMDB."""

    manifest = load_poster_manifest()
    key = _manifest_key(movie_id)
    cached = manifest.get(key)

    if isinstance(cached, dict) and cached.get("poster_url"):
        return cached

    visual = _fetch_visual_from_tmdb(movie_id, title, api_key)

    if persist and api_key:
        with _MANIFEST_LOCK:
            latest = dict(load_poster_manifest())
            latest[key] = visual

            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            temporary = POSTER_METADATA_PATH.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(latest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, POSTER_METADATA_PATH)

            load_poster_manifest.cache_clear()

    return visual


def get_movie_visuals_batch(
    movies: Iterable[dict[str, Any]],
    api_key: str | None,
    *,
    max_workers: int = 8,
    persist: bool = True,
) -> dict[int, dict[str, Any]]:
    """Resolve visible posters concurrently instead of one-by-one."""

    movie_list = [
        {
            "id": int(movie["id"]),
            "title": str(movie["title"]),
        }
        for movie in movies
    ]

    manifest = load_poster_manifest()
    results: dict[int, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []

    for movie in movie_list:
        cached = manifest.get(_manifest_key(movie["id"]))

        if isinstance(cached, dict) and cached.get("poster_url"):
            results[movie["id"]] = cached
        else:
            missing.append(movie)

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            future_map = {
                executor.submit(
                    _fetch_visual_from_tmdb,
                    movie["id"],
                    movie["title"],
                    api_key,
                ): movie
                for movie in missing
            }

            fetched: dict[str, dict[str, Any]] = {}

            for future in as_completed(future_map):
                movie = future_map[future]

                try:
                    visual = future.result()
                except Exception:
                    visual = _normalise_visual(
                        movie["id"],
                        movie["title"],
                        {},
                    )

                results[movie["id"]] = visual
                fetched[_manifest_key(movie["id"])] = visual

        if persist and api_key and fetched:
            with _MANIFEST_LOCK:
                latest = dict(load_poster_manifest())
                latest.update(fetched)

                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                temporary = POSTER_METADATA_PATH.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(latest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary, POSTER_METADATA_PATH)

                load_poster_manifest.cache_clear()

    return results


def build_full_poster_manifest(
    engine: dict[str, Any],
    api_key: str,
    *,
    max_workers: int = 6,
    batch_size: int = 100,
) -> dict[str, dict[str, Any]]:
    """Build poster metadata for the full catalog before deployment."""

    if not api_key:
        raise RecommenderError("TMDB_API_KEY is required to build poster metadata.")

    movies = engine["movies"][["id", "title"]].to_dict(orient="records")
    manifest = dict(load_poster_manifest())

    missing = [
        movie
        for movie in movies
        if _manifest_key(int(movie["id"])) not in manifest
    ]

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        fetched = get_movie_visuals_batch(
            batch,
            api_key,
            max_workers=max_workers,
            persist=False,
        )

        for movie_id, visual in fetched.items():
            manifest[_manifest_key(movie_id)] = visual

        save_poster_manifest(manifest)

        completed = min(start + len(batch), len(missing))
        print(
            f"Poster metadata: {completed:,}/{len(missing):,} missing movies processed",
            flush=True,
        )

    load_poster_manifest.cache_clear()
    return manifest


def get_poster_url(
    movie_id: int,
    title: str,
    api_key: str | None,
) -> str:
    return str(
        get_movie_visual(movie_id, title, api_key).get(
            "poster_url",
            _placeholder_url(title),
        )
    )


def get_movie_details(
    movie_id: int,
    api_key: str | None,
    title: str = "",
) -> dict[str, Any]:
    visual = get_movie_visual(
        movie_id,
        title or f"Movie {movie_id}",
        api_key,
    )

    return {
        "year": visual.get("year"),
        "rating": visual.get("rating"),
        "overview": visual.get("overview"),
    }