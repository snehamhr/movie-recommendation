# Movie Recommendation System

A Streamlit content-based movie recommendation application built from the TMDB 5000 Movie Metadata dataset.

## Project intent

The project helps users discover movies similar to a title they already like. It compares movie content—genres, overview, keywords, top cast members and director—and returns the five closest matches.

Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Recommendation logic

The recommendation flow preserves the uploaded notebook approach:

1. Merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on `title`.
2. Keep `title`, `genres`, `id`, `overview`, `keywords`, `cast` and `crew`.
3. Remove rows with missing required data.
4. Extract all genres and keywords.
5. Keep the first three cast members.
6. Keep the first director.
7. Remove spaces inside multi-word names.
8. Combine the fields into a single `tags` column.
9. Convert tags to lowercase and apply Porter stemming.
10. Use `CountVectorizer(max_features=5000, stop_words="english")`.
11. Rank movies using cosine similarity.
12. Return the top five titles excluding the selected movie.

The implementation fixes two notebook execution mistakes without changing the intended method: stemming is applied before vectorization, and recommendations use the selected movie's similarity scores.

## Features

- Searchable movie selector
- Five content-based recommendations
- Movie poster cards
- Similarity percentage
- Optional release year and TMDB rating
- Automatic Kaggle dataset download when local CSV files are absent
- Cached recommendation artifact for faster future starts

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app automatically downloads and processes the public Kaggle dataset when the saved artifact and local CSV files are absent.

## Official posters

Create a free TMDB API key and add it to `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

Without a key, the recommendation engine still works and displays title-based poster placeholders.

## Project structure

```text
movie-recommendation-streamlit/
├── app.py
├── recommender.py
├── build_artifacts.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── artifacts/
│   └── .gitkeep
└── data/
    └── .gitkeep
```


## Catalog interface

The Streamlit interface presents the complete movie catalog through paginated poster cards.

- 24 movies are shown per page.
- Users can navigate with Previous, Next, or a direct page number.
- Search filters the full catalog.
- Each card includes a poster, title, year/rating when available, and a Select button.
- The chosen movie appears in a dedicated selected-movie panel.
- Clicking Get recommendations displays the top five matches with posters and cosine-similarity scores.

The recommendation model itself remains unchanged from the notebook workflow.
