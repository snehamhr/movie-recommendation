from recommender import ARTIFACT_PATH, build_recommender

if __name__ == "__main__":
    bundle = build_recommender()
    print(f"Saved artifact: {ARTIFACT_PATH}")
    print(f"Movies indexed: {len(bundle['movies']):,}")
