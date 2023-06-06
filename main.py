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


# Define the inference endpoint
@app.post("/inference")
def inference(request: InferenceRequest):
    new_user_items = request.new_user_items
   
    # Perform inference
    recommendations = run_inference_on_new_user(
        items, data_tfidf, k=10, new_user_items=new_user_items
    )
    
    return recommendations.get_html_string()

# Run the FastAPI app
if __name__ == "__main__":
    

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
