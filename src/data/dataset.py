from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.utils.logger import get_logger
import torch
import os

logger = get_logger(__file__)

class TripletDataset(Dataset):
    def __init__(self, encodings):
        self.anchor_ids = encodings['anchor_input_ids']
        self.anchor_mask = encodings['anchor_attention_mask']
        self.positive_ids = encodings['positive_input_ids']
        self.positive_mask = encodings['positive_attention_mask']
        self.negative_ids = encodings['negative_input_ids']
        self.negative_mask = encodings['negative_attention_mask']

    def __len__(self):
        return len(self.anchor_ids)

    def __getitem__(self, idx):
        return {
            'anchor': {
                'input_ids': self.anchor_ids[idx],
                'attention_mask': self.anchor_mask[idx],
            },
            'positive': {
                'input_ids': self.positive_ids[idx],
                'attention_mask': self.positive_mask[idx],
            },
            'negative': {
                'input_ids': self.negative_ids[idx],
                'attention_mask': self.negative_mask[idx],
            },
        }


def triplet_collate_fn(batch):
    result = {}

    for key in ['anchor', 'positive', 'negative']:
        input_ids = [sample[key]['input_ids'] for sample in batch]
        attention_masks = [sample[key]['attention_mask'] for sample in batch]

        #pad_sequence pads to the longest sequence in this batch
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        result[key] = {
            'input_ids': padded_ids,
            'attention_mask': padded_masks,
        }

    return result


def get_dataloaders(batch_size=32):
    base_path = os.path.join(os.path.dirname(__file__), '../../data/processed')

    dataloaders = {}

    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split_name, 'encodings.pt')
        encodings = torch.load(split_path, weights_only=False)
        dataset = TripletDataset(encodings)

        shuffle = (split_name == 'train')
        dataloaders[split_name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=triplet_collate_fn
        )

        logger.info(f'Created {split_name} dataloader: {len(dataset)} samples, batch_size={batch_size}')

    return dataloaders['train'], dataloaders['val'], dataloaders['test']
