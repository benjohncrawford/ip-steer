import pandas as pd


from ipsteer.utils import get_project_dir
from openai import OpenAI

data_dir = get_project_dir() / 'data' / 'toxicity'
jigsaw_dir = data_dir / 'jigsaw'

df = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines = True, orient = 'records')
client = OpenAI()
for row in df.iterrows(index=True):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=row.text,
    ).to_dict()

    # Response has several results including a boolean flag and dict of category scores
    scores = response["results"][0]["category_scores"]
    row["scores"] = scores
    
df.to_json(jigsaw_dir / 'final_train_with_scores.jsonl', lines = True, orient = 'records')
    

   