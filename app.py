from __future__ import annotations

import html
import math
import os
from typing import Any

import streamlit as st

from recommender import (
    RecommenderError,
    ensure_recommender,
    get_movie_visuals_batch,
    recommend_movies,
)

st.set_page_config(
    page_title="CineMatch — Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOVIES_PER_PAGE = 24
GRID_COLUMNS = 6

CUSTOM_CSS = """
<style>
:root {
    --line: rgba(255,255,255,.09);
    --text: #f7f7fb;
    --muted: #a7adbd;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(139,92,246,.18), transparent 26rem),
        radial-gradient(circle at 85% 5%, rgba(229,9,20,.16), transparent 30rem),
        linear-gradient(180deg, #090a0f 0%, #08090d 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 4rem;
    max-width: 1550px;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(139,92,246,.18), transparent 18rem),
        linear-gradient(180deg, #0d0f16 0%, #090a0f 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem;
}

.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    margin-bottom: 1.25rem;
    background:
        linear-gradient(115deg, rgba(14,16,24,.98), rgba(43,18,57,.94)),
        radial-gradient(circle at 90% 20%, rgba(229,9,20,.25), transparent 20rem);
    border: 1px solid var(--line);
    box-shadow: 0 24px 70px rgba(0,0,0,.35);
}

.hero h1 {
    margin: 0;
    font-size: 2.75rem;
    letter-spacing: -.045em;
}

.hero p {
    margin: .65rem 0 0;
    max-width: 900px;
    color: #c0c5d2;
    font-size: 1.02rem;
}

.section-header {
    margin: 1.35rem 0 .85rem;
}

.section-header h2 {
    margin: 0;
    font-size: 1.55rem;
}

.section-header p {
    margin: .2rem 0 0;
    color: var(--muted);
    font-size: .9rem;
}

.movie-title {
    font-weight: 750;
    font-size: .92rem;
    line-height: 1.28;
    min-height: 2.35rem;
    margin: .5rem .15rem .2rem;
}

.movie-meta {
    color: var(--muted);
    font-size: .76rem;
    margin: 0 .15rem .35rem;
}

.sidebar-movie-title {
    font-size: 1.08rem;
    font-weight: 800;
    line-height: 1.3;
    margin-top: .45rem;
}

.sidebar-section-title {
    font-size: 1.05rem;
    font-weight: 800;
    margin: 1rem 0 .55rem;
}

.sidebar-rec-title {
    font-size: .84rem;
    font-weight: 740;
    line-height: 1.25;
    margin-top: .3rem;
}

.sidebar-similarity {
    color: #d8b4fe;
    font-size: .75rem;
    font-weight: 700;
    margin-bottom: .65rem;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    aspect-ratio: 2 / 3;
    object-fit: cover;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 730;
    border: 1px solid rgba(255,255,255,.10);
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    border-radius: 12px;
}

.page-note,
.footer-note {
    color: var(--muted);
    font-size: .84rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _secret_or_env(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name)


@st.cache_resource(show_spinner="Preparing the movie recommendation engine...")
def load_engine() -> dict[str, Any]:
    return ensure_recommender()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_visuals_for_movies(
    movie_records: tuple[tuple[int, str], ...],
    api_key: str | None,
) -> dict[int, dict[str, Any]]:
    movies = [
        {
            "id": movie_id,
            "title": title,
        }
        for movie_id, title in movie_records
    ]

    return get_movie_visuals_batch(
        movies,
        api_key,
        max_workers=8,
    )


def visual_map_for_frame(
    frame,
    api_key: str | None,
) -> dict[int, dict[str, Any]]:
    records = tuple(
        (int(row["id"]), str(row["title"]))
        for _, row in frame.iterrows()
    )

    return load_visuals_for_movies(records, api_key)


def render_catalog_card(
    movie: dict[str, Any],
    visual: dict[str, Any],
    position: int,
    engine: dict[str, Any],
) -> None:
    movie_id = int(movie["id"])
    title = str(movie["title"])

    st.image(
        visual["poster_url"],
        use_container_width=True,
    )

    st.markdown(
        f"<div class='movie-title'>{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )

    meta_parts: list[str] = []

    if visual.get("year"):
        meta_parts.append(str(visual["year"]))

    if visual.get("rating") is not None:
        meta_parts.append(f"⭐ {float(visual['rating']):.1f}")

    st.markdown(
        f"<div class='movie-meta'>"
        f"{html.escape(' · '.join(meta_parts) or 'TMDB movie')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "Select",
        key=f"select_{movie_id}_{position}",
        use_container_width=True,
    ):
        try:
            recommendations = recommend_movies(
                movie_title=title,
                engine=engine,
                top_n=5,
            )
        except RecommenderError as exc:
            st.error(str(exc))
        else:
            st.session_state["selected_movie"] = title
            st.session_state["recommendations"] = recommendations
            st.rerun()


def render_sidebar_results(
    engine: dict[str, Any],
    api_key: str | None,
) -> None:
    selected_title = st.session_state.get("selected_movie")
    recommendations = st.session_state.get("recommendations", [])

    with st.sidebar:
        st.markdown("## 🎞️ Your selection")

        if not selected_title:
            st.info(
                "Select any movie from the catalog to see its"
                "top five recommendations here."
            )
            return

        selected_rows = engine["movies"].loc[
            engine["movies"]["title"].eq(selected_title)
        ]

        if selected_rows.empty:
            st.warning("The selected movie is no longer available.")
            st.session_state.pop("selected_movie", None)
            st.session_state.pop("recommendations", None)
            return

        selected_row = selected_rows.iloc[0]

        sidebar_movies = [
            {
                "id": int(selected_row["id"]),
                "title": selected_title,
            },
            *[
                {
                    "id": int(movie["id"]),
                    "title": str(movie["title"]),
                }
                for movie in recommendations
            ],
        ]

        visuals = get_movie_visuals_batch(
            sidebar_movies,
            api_key,
            max_workers=6,
        )

        selected_id = int(selected_row["id"])
        selected_visual = visuals[selected_id]

        st.image(
            selected_visual["poster_url"],
            use_container_width=True,
        )

        st.markdown(
            f"<div class='sidebar-movie-title'>"
            f"{html.escape(selected_title)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        meta_parts: list[str] = []

        if selected_visual.get("year"):
            meta_parts.append(str(selected_visual["year"]))

        if selected_visual.get("rating") is not None:
            meta_parts.append(
                f"⭐ {float(selected_visual['rating']):.1f}"
            )

        if meta_parts:
            st.caption(" · ".join(meta_parts))

        st.markdown(
            "<div class='sidebar-section-title'>Top 5 recommendations</div>",
            unsafe_allow_html=True,
        )

        for movie in recommendations:
            movie_id = int(movie["id"])
            title = str(movie["title"])
            visual = visuals[movie_id]

            poster_column, text_column = st.columns(
                [1, 1.45],
                vertical_alignment="center",
            )

            with poster_column:
                st.image(
                    visual["poster_url"],
                    use_container_width=True,
                )

            with text_column:
                st.markdown(
                    f"<div class='sidebar-rec-title'>"
                    f"{html.escape(title)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='sidebar-similarity'>"
                    f"{float(movie['similarity']) * 100:.1f}% similarity"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🎬 CineMatch</h1>
            <p>
                Browse the complete TMDB movie catalog, select a movie you like,
                and discover five content-based recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        engine = load_engine()
    except RecommenderError as exc:
        st.error(str(exc))
        st.stop()

    api_key = _secret_or_env("TMDB_API_KEY")

    render_sidebar_results(
        engine=engine,
        api_key=api_key,
    )

    all_movies = engine["movies"].copy()

    search_column, info_column = st.columns(
        [3, 1],
        vertical_alignment="bottom",
    )

    with search_column:
        search_text = st.text_input(
            "Search movies",
            placeholder="Type a movie name...",
        ).strip()

    with info_column:
        st.metric(
            "Movies available",
            f"{len(all_movies):,}",
        )

    previous_search = st.session_state.get("previous_search_text", "")

    if search_text != previous_search:
        st.session_state["previous_search_text"] = search_text
        st.session_state["catalog_page"] = 1

    if search_text:
        filtered = all_movies[
            all_movies["title"].str.contains(
                search_text,
                case=False,
                na=False,
                regex=False,
            )
        ].reset_index(drop=True)
    else:
        filtered = all_movies.reset_index(drop=True)

    total_movies = len(filtered)
    total_pages = max(1, math.ceil(total_movies / MOVIES_PER_PAGE))

    st.session_state.setdefault("catalog_page", 1)
    st.session_state["catalog_page"] = min(
        max(int(st.session_state["catalog_page"]), 1),
        total_pages,
    )

    previous_column, page_input_column, page_info_column, next_column = st.columns(
        [1, 1.5, 1.5, 1]
    )

    with previous_column:
        if st.button(
            "← Previous",
            disabled=st.session_state["catalog_page"] <= 1,
            use_container_width=True,
        ):
            st.session_state["catalog_page"] -= 1
            st.rerun()

    with page_input_column:
        chosen_page = st.number_input(
            "Go to page",
            min_value=1,
            max_value=total_pages,
            value=int(st.session_state["catalog_page"]),
            step=1,
        )

        if int(chosen_page) != st.session_state["catalog_page"]:
            st.session_state["catalog_page"] = int(chosen_page)
            st.rerun()

    with page_info_column:
        st.markdown(
            f"<div class='page-note' style='padding-top:2rem'>"
            f"Page <b>{st.session_state['catalog_page']}</b> "
            f"of <b>{total_pages}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with next_column:
        if st.button(
            "Next →",
            disabled=st.session_state["catalog_page"] >= total_pages,
            use_container_width=True,
        ):
            st.session_state["catalog_page"] += 1
            st.rerun()

    start = (st.session_state["catalog_page"] - 1) * MOVIES_PER_PAGE
    end = min(start + MOVIES_PER_PAGE, total_movies)
    page_movies = filtered.iloc[start:end]

    st.markdown(
        f"""
        <div class="section-header">
            <h2>Browse movies</h2>
            <p>
                Showing {start + 1 if total_movies else 0}–{end}
                of {total_movies:,} movies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page_movies.empty:
        st.warning("No movies matched your search.")
    else:
        with st.spinner("Loading movie posters..."):
            page_visuals = visual_map_for_frame(
                page_movies,
                api_key,
            )

        for row_start in range(0, len(page_movies), GRID_COLUMNS):
            columns = st.columns(GRID_COLUMNS)
            chunk = page_movies.iloc[
                row_start : row_start + GRID_COLUMNS
            ]

            for offset, (_, movie) in enumerate(chunk.iterrows()):
                movie_dict = movie.to_dict()
                movie_id = int(movie_dict["id"])

                with columns[offset]:
                    render_catalog_card(
                        movie=movie_dict,
                        visual=page_visuals[movie_id],
                        position=start + row_start + offset,
                        engine=engine,
                    )

    st.markdown(
        "<div class='footer-note'>"
        "For the fastest deployment, commit "
        "`artifacts/poster_metadata.json` after building it locally."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()