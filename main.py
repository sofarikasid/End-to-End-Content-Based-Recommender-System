from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from mylib.run_rec_sys import run_inference_on_new_user


# Define the input data model
class InferenceRequest(BaseModel):
    new_user_items: list[int]


# Load the necessary data for inference
data_tfidf = pd.read_csv("mylib/data_tfidf_stem_test.csv", index_col="article_id")

items = pd.read_csv("mylib/articles.csv", index_col="article_id")

user_profile = pd.read_csv("mylib/user_profile_test.csv", index_col="article_id").drop(
    "Unnamed: 0", axis=1
)


# Create the FastAPI app
app = FastAPI()


@app.get("/")
async def read_root():
    """Home Page"""
    return {"Content": "Welcome to the Recommender System API"}


# Define the inference route that accepts the input data model and k as a query parameter and returns the recommended items as a personalized list
@app.post("/inference")
async def inference(inference_request: InferenceRequest, k: int = 10):
    """Inference Page"""
    # Get the new user items
    new_user_items = inference_request.new_user_items

    # Run inference on the new user
    recommended_items = run_inference_on_new_user(items, data_tfidf, k, new_user_items)

    # it recieved a list of items and returns a list of recommended items
    out_recommendation_data = items.iloc[recommended_items, [1, 20]].copy()

    # output is a df with the recommended items
    return out_recommendation_data.to_dict(orient="records")


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
