"""Run the full local pipeline from clean data through artifacts."""

from src import data_cleaning, eda, evaluate, explainability, feature_engineering, insights, sql_analysis, statistics, train


def main() -> None:
    data_cleaning.main()
    feature_engineering.main()
    eda.main()
    statistics.main()
    sql_analysis.main()
    train.main()
    evaluate.main()
    explainability.main()
    insights.main()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
