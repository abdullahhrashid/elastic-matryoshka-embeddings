from src.data.prepare_dataset import tokenize_and_save
from src.utils.config import load_config
from src.utils.logger import get_logger
import argparse

logger = get_logger(__file__)

parser = argparse.ArgumentParser(description='Tokenize and save the dataset')
parser.add_argument('--config', type=str, default=None, help='path to config file')

args = parser.parse_args()
config = load_config(args.config)

tokenize_and_save(config)

logger.info('Dataset tokenized and saved')
