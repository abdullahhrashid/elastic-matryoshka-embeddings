from datasets import load_dataset, concatenate_datasets, Dataset
from dotenv import load_dotenv
from src.utils.logger import get_logger
import random
import os

#loading the hf hub token
load_dotenv()

logger = get_logger(__file__)

#because this dataset is too big, i will stream it
def load_trim_shuffle(split_name, n_samples, seed=42):
    ds_stream = load_dataset('thebajajra/hard-negative-triplets', split=split_name, streaming=True)
    
    unique_data = []
    seen_anchors = set()
        
    for item in ds_stream:
        anchor_text = item['anchor']
        
        #only keeping the row if we haven't seen this anchor yet
        if anchor_text not in seen_anchors:
            seen_anchors.add(anchor_text)
            unique_data.append(item)
            
        #stop completely once we have exactly what we need
        if len(unique_data) == n_samples:
            break

    #shuffling the resulting unique data 
    split_seed = seed + hash(split_name) % (2**32)
    rng = random.Random(split_seed)
    rng.shuffle(unique_data)

    ds = Dataset.from_list(unique_data)

    #unneeded column
    ds = ds.remove_columns(['dataset'])

    return ds

#a gold standard dataset for embedding models
msmarco = load_trim_shuffle('msmarco_distillation_simlm_rescored_reranked_min15', 80_000)
logger.info('Loaded MS Marco Triples')

nq = load_trim_shuffle('nq_cocondensor_hn_mine_reranked_min15', 50_000)
logger.info('Loaded NQ Triples')

#a good dataset with sequences of long lengths
hotpotqa = load_trim_shuffle('hotpotqa_hn_mine_shuffled', 50_000)
logger.info('Loaded HotPotQA Triples')

combined = concatenate_datasets([msmarco, nq, hotpotqa, reddit]).shuffle(seed=42)
logger.info('Combined all shuffled datasets')

path = os.path.join(os.path.dirname(__file__), '../../data/sequences')

#saving for when we fine tune our embedding model
combined.save_to_disk(path)
logger.info('Saved combined dataset')
