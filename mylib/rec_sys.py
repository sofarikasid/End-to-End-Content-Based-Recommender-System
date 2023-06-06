#!/usr/bin/env python
import numpy as np


class ContentBasedRS:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def compute_recommendation(self, user_previous_purchase):
        "This method computes a content-based recommendation for a given user"
        Data_ = np.array(self.data)
        user_cenriod = self.data.loc[user_previous_purchase].mean()
        D_norm = np.array([np.linalg.norm(Data_[i]) for i in range(len(Data_))])
        x_norm = np.linalg.norm(user_cenriod)
        sims = np.dot(Data_, user_cenriod) / (D_norm * x_norm)
        dists = 1 - sims
        idx = np.argsort(dists)
        return idx[: self.k], sims[idx][: self.k]


class Recommender:
    def __init__(self, user_profile, data, k):
        self.user_profile = user_profile
        self.data = data
        self.k = k
        self.recommendations = []
        self.cosine_sims = []

    def run_recommendation(self):
        "This method runs the recommendation on all users in the user_profile"
        for user in range(len(self.user_profile)):
            rs = ContentBasedRS(self.data, self.k)
            similar_songs, cosine_sim = rs.compute_recommendation(
                self.user_profile[user]
            )
            self.recommendations.append(similar_songs)
            self.cosine_sims.append(cosine_sim)

    def calculate_recall_precision(self, relevance_user_purchases):
        "This method calculates recall and precision for the recommendations"
        recalls = []
        precisions = []
        for i in range(len(self.recommendations)):
            recs = self.recommendations[i]
            rels = relevance_user_purchases[i]
            common_items = set(recs).intersection(set(rels))
            recall = len(common_items) / len(rels)
            precision = len(common_items) / len(recs) if len(recs) > 0 else 0
            recalls.append(recall)
            precisions.append(precision)
        return np.mean(recalls), np.mean(precisions)
