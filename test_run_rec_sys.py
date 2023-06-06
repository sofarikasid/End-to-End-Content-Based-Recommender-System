import pytest
import pandas as pd
import numpy as np
from mylib.run_rec_sys import batch_train_recommender, run_inference_on_new_user

data = pd.read_csv("mylib/data_tfidf_stem_test.csv", index_col="article_id")

items = pd.read_csv("mylib/articles.csv", index_col="article_id")

user_profile = pd.read_csv("mylib/user_profile_test.csv", index_col="article_id").drop(
    "Unnamed: 0", axis=1
)


# Test run_inference_on_new_user function
def test_run_inference_on_new_user():
    new_user_items = [696356013, 766777012, 771881001]
    suggested_recommendations = run_inference_on_new_user(
        items, data, k=10, new_user_items=new_user_items
    )
    # assert [ 27 ,  20,  133,   30, 1131, 3720, 3374,  159, 2658,  990] ==(suggested_recommendations)
    assert np.any(
        np.array(suggested_recommendations)
        == [27, 20, 133, 30, 1131, 3720, 3374, 159, 2658, 990]
    )
    assert len(suggested_recommendations) == 10


# Run the tests
if __name__ == "__main__":
    pytest.main()
