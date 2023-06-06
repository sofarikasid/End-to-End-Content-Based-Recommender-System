#!/usr/bin/env python
import fire
import pandas as pd
from mylib.run_rec_sys import batch_train_recommender, run_inference_on_new_user


def open_csv_file(tfidf_file, items_file, user_profile_file):
    """THIA FUNCTION OPENS THE CSV FILES AND RETURNS THE DATAFRAMES"""
    # Read the CSV files using pandas
    data_tfidf = pd.read_csv(tfidf_file, index_col="article_id")
    items_ = pd.read_csv(items_file, index_col="article_id")
    user_profile = pd.read_csv(user_profile_file, index_col="article_id").drop(
        "Unnamed: 0", axis=1
    )

    return data_tfidf, items_, user_profile


def run_recommender_cli(
    tfidf_file, items_file, user_profile_file, k=10, batch_size=1000
):
    """
    This function trains a recommender on a batch of users and returns k recommendations.
    """
    data_tfidf, items_, user_profile = open_csv_file(
        tfidf_file, items_file, user_profile_file
    )

    recommender = batch_train_recommender(data_tfidf, user_profile, k, batch_size)
    print(f"The system is trained on {batch_size} users")
    return recommender


def run_inference_cli(
    tfidf_file, items_file, user_profile_file, k=10, new_user_items=None
):
    """
    This function runs inference on a new user by taking a list of items purchased by the user.
    """
    data_tfidf, items_, user_profile = open_csv_file(
        tfidf_file, items_file, user_profile_file
    )

    recommendations = run_inference_on_new_user(items_, data_tfidf, k, new_user_items)
    
    return recommendations


if __name__ == "__main__":
    fire.Fire(run_inference_cli)
