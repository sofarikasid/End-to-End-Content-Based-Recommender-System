#!/usr/bin/env python

import pandas as pd
import json
import numpy as np
from rec_sys import Recommender, ContentBasedRS


## Load data
# data_ = pd.read_csv("item_descr_train_doc2vec_2.csv", index_col="article_id")
# data_tfidf_stem_ = pd.read_csv("mylib/tfidf_stem_desc.csv", index_col="article_id")
# user_profile_ = pd.read_csv("mylib/user_profile_train_and_relevant.csv").drop("Unnamed: 0", axis=1)


def batch_train_recommender(data, user_profile, k=10, batch_size=1000):
    """This function trains a recommender on a batch of users and returns k recommendations
    if K and batch_size are not specified, the default values are 10 and 1000 respectively
    """

    batch_input = user_profile[:batch_size]
    train_data = np.array(batch_input["user_train_items"])
    relevance_data = np.array(batch_input["relevant_items"])

    relevance_user_purchases = []
    user_purchases = []

    for items_purchased in range(len(train_data)):
        lst = json.loads(items_purchased)
        user_purchases.append(lst)

    for relevant in range(len(relevance_data)):
        lst = json.loads(relevant)
        relevance_user_purchases.append(lst)

    recommender = Recommender(user_purchases, data, k)
    recommender.run_recommendation()
    return recommender


def run_inference_on_new_user(data, k=10, new_user_items=None):
    """
    This function runs inference on a new user by taking a list of items purchased by the user
    """

    # Create an instance of the ContentBasedRS class
    rs = ContentBasedRS(data, k)

    # compute the recommendations for the new user
    suggested_recommendations, cosine_sim = rs.compute_recommendation(new_user_items)

    ########
    items = pd.read_csv("articles.csv", index_col="article_id")
    recomendatio_item = items.iloc[suggested_recommendations, [1, 20]]
    recomendatio_item["cosine_sim"] = cosine_sim

    return recomendatio_item
