import argparse
from tqdm import trange

import pandas as pd
import torch

from ipsteer.utils import get_project_dir
from ipsteer.lm import HuggingFaceLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type = str, default = 'Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx', type = int, default = -1)
    parser.add_argument('-b', '--batch_size', type = int, default = 10)
    args = parser.parse_args()
    model_name = ""
    batch_size = 10
    layer_idx = 10
    
    def safe_len_ratio(arr_1, arr_2):
        if len(arr_1) and len(arr_2):
            return len(arr_1) / len(arr_2)
        return len(arr_1) / 1

    # extract activations
    data_dir = get_project_dir() / 'data' / 'toxicity'
    jigsaw_dir = data_dir / 'jigsaw'
    activations_dir = data_dir / 'activations' / model_name
    activations_dir.mkdir(parents = True, exist_ok = True)

    model = HuggingFaceLM(model_name, device = "auto", dtype = torch.float32)
    layer_idx = model.steer_layer_idx if layer_idx == -1 else layer_idx

    final_train_df = pd.read_json(jigsaw_dir / 'final_train_with_scores_first_6000.jsonl', lines = True, orient = 'records')
    num_batches = (len(final_train_df) + batch_size - 1) // batch_size


    objectives = ["hate", "violence"]
    pos_activations = {}
    neg_activations = {}
    for objective in objectives:
        pos_activations[objective] = []
        neg_activations[objective] = []
    batch_pos_texts = {}
    batch_neg_texts = {}    
    for i in trange(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(final_train_df))
        batch_data = final_train_df.iloc[start_idx:end_idx]

        for objective in objectives:
            batch_pos_texts[objective] = []
            batch_neg_texts[objective] = []
                
        for idx, label in enumerate(final_train_df[objective].iloc[start_idx:end_idx]):
            for objective in objectives: 
                # undersample positive activations so we don't have too imbalanced classes
                if label <= 0.5 and safe_len_ratio(pos_activations[objective], neg_activations[objective]) <= 5:
                    batch_pos_texts[objective].append(batch_data['text'].iloc[idx])
                elif label > 0.5:
                    batch_neg_texts[objective].append(batch_data['text'].iloc[idx])
            
                if len(batch_pos_texts) > 0:
                    pos_activations[objective].append(model.extract_prompt_eos_activations(batch_pos_texts[objective], layer_idx = layer_idx).cpu())
                if len(batch_neg_texts) > 0:
                    neg_activations[objective].append(model.extract_prompt_eos_activations(batch_neg_texts[objective], layer_idx = layer_idx).cpu())
                    
    for objective in objectives:        
        pos_activations = torch.cat(pos_activations[objective], dim = 0)
        neg_activations = torch.cat(neg_activations[objective], dim = 0)
        torch.save(pos_activations, activations_dir / f'jigsaw_pos_activations_layer{layer_idx}_objective_{objective}.pt')
        torch.save(neg_activations, activations_dir / f'jigsaw_neg_activations_layer{layer_idx}_objective_{objective}.pt')