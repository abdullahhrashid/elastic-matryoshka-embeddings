from datasets import load_from_disk
from transformers import AutoTokenizer
from src.utils.logger import get_logger
import torch
import os

logger = get_logger(__file__)

def tokenize_and_save(config):
    raw_path = os.path.join(os.path.dirname(__file__), '../../', config['data']['raw_path'])
    save_path = os.path.join(os.path.dirname(__file__), '../../', config['data']['processed_path'])

    max_length = config['data']['max_length']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    seed = config['data']['seed']

    ds = load_from_disk(raw_path)
    logger.info(f'Loaded {len(ds)} raw triplets')

    #splitting into train, val and test
    ds = ds.train_test_split(test_size=(1 - train_ratio), seed=seed)
    train_ds = ds['train']
    remaining = ds['test']

    val_test_ratio = val_ratio / (1 - train_ratio)
    remaining = remaining.train_test_split(test_size=(1 - val_test_ratio), seed=seed)
    val_ds = remaining['train']
    test_ds = remaining['test']

    logger.info(f'Split the dataset into - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}')

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True, use_fast=True)

    def tokenize_split(split_ds, split_name):
        logger.info(f'Tokenizing {split_name} split ({len(split_ds)} samples)')

        #tokenizing without padding, just truncation. this keeps each sequence at its natural length
        anchor_enc = tokenizer(
            list(split_ds['anchor']), max_length=max_length, truncation=True
        )
        positive_enc = tokenizer(
            list(split_ds['positive']), max_length=max_length, truncation=True
        )
        negative_enc = tokenizer(
            list(split_ds['negative']), max_length=max_length, truncation=True
        )

        #saving as lists of variable length tensors, more memory efficient
        encodings = {
            'anchor_input_ids': [torch.tensor(ids) for ids in anchor_enc['input_ids']],
            'anchor_attention_mask': [torch.tensor(mask) for ids, mask in zip(anchor_enc['input_ids'], anchor_enc['attention_mask'])],
            'positive_input_ids': [torch.tensor(ids) for ids in positive_enc['input_ids']],
            'positive_attention_mask': [torch.tensor(mask) for ids, mask in zip(positive_enc['input_ids'], positive_enc['attention_mask'])],
            'negative_input_ids': [torch.tensor(ids) for ids in negative_enc['input_ids']],
            'negative_attention_mask': [torch.tensor(mask) for ids, mask in zip(negative_enc['input_ids'], negative_enc['attention_mask'])],
        }

        split_path = os.path.join(save_path, split_name)
        os.makedirs(split_path, exist_ok=True)
        torch.save(encodings, os.path.join(split_path, 'encodings.pt'))
        logger.info(f'Saved tokenized {split_name} to {split_path}')

    tokenize_split(train_ds, 'train')
    tokenize_split(val_ds, 'val')
    tokenize_split(test_ds, 'test')

    logger.info('All splits tokenized and saved')
