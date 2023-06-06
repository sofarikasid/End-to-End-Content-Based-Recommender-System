import fire

from mylib.run_rec_sys import batch_train_recommender, run_inference_on_new_user


# Write a CLI function to run the recommender on a batch of users, allowing the user to specify the batch size and the number of recommendations
def run_recommender_cli(data, user_profile, k=10, batch_size=1000):
    recommender = batch_train_recommender(data, user_profile, k, batch_size)
    print(f"The system is trained on {batch_size} users")
    return recommender


# Write a CLI function to run inference on a new user
def run_inference_cli(data, k=10, new_user_items=None):
    recommendations = run_inference_on_new_user(data, k, new_user_items)
    print(f"This is the recommendation for the new user: {recommendations}")
    return recommendations


if __name__ == "__main__":
    fire.Fire(run_inference_cli)

if __name__ == "__main__":
    fire.Fire(run_recommender_cli)
