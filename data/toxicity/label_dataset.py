import pandas as pd


from ipsteer.utils import get_project_dir
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from tenacity import (
retry,
stop_after_attempt,
wait_random_exponential,
) # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def moderation_with_backoff(**kwargs):
    return client.moderations.create(**kwargs)

load_dotenv()
data_dir = get_project_dir() / 'data' / 'toxicity'
jigsaw_dir = data_dir / 'jigsaw'

df = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines = True, orient = 'records')
client = OpenAI()


categories = ['harassment', 'harassment/threatening', 'hate', 'hate/threatening', 'illicit', 'illicit/violent', 'self-harm', 'self-harm/instructions', 'self-harm/intent', 'sexual', 'sexual/minors', 'violence', 'violence/graphic']
result_dict = {}
for category in categories:
    result_dict[category] = [0]*len(df.index)

save_after = 1000
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    response = moderation_with_backoff(
        model="omni-moderation-latest",
        input=row.text,
    )
    response = response.to_dict()
    # Response has several results including a boolean flag and dict of category scores
    scores = response["results"][0]["category_scores"]
    for category in categories:
        result_dict[category][index] = scores[category]
    
    if index % save_after == 0: 
        for category, category_scores in result_dict.items():
            df[category] = category_scores
        df.to_json(jigsaw_dir / 'final_train_with_scores.jsonl', lines = True, orient = 'records')
        
print("Finished saving results...")
for category, category_scores in result_dict.items():
    df[category] = category_scores
df.to_json(jigsaw_dir / 'final_train_with_scores.jsonl', lines = True, orient = 'records')