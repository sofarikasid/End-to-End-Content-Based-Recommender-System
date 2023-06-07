 import pandas as pd

from mylib.run_rec_sys import batch_train_recommender, run_inference_on_new_user

data_tfidf = pd.read_csv("mylib/data_tfidf_stem_test.csv", index_col="article_id")

items_ = pd.read_csv("mylib/articles.csv", index_col="article_id")

user_profle= pd.read_csv("mylib/user_profile_test.csv", index_col="article_id").drop('Unnamed: 0', axis=1)

run_inference_on_new_user(items_,data_tfidf,5,[663713001])
run_inference_on_new_user(items_,data_tfidf,5,[663713001,487205007])
run_inference_on_new_user(items_,data_tfidf,5,[663713001,487205007,770587002])



#### Command Line Prompt to run  
chmod +x CLI_run_rec_sys.py
./CLI_run_rec_sys.py --tfidf_file=mylib/data_tfidf_stem_test.csv --items_file=mylib/articles.csv --user_profile_file=mylib/user_profile_test.csv --k 10 --new_user_items [696356013]
./CLI_run_rec_sys.py --tfidf_file=mylib/data_tfidf_stem_test.csv --items_file=mylib/articles.csv --user_profile_file=mylib/user_profile_test.csv --k 10 --new_user_items [696356013,766777012,771881001]
########################################################
### BUILD AND RUN docker
# Build the Docker image
docker build -t your-image-name .


# Run the Docker container
docker run -p 8000:8000 your-image-name


