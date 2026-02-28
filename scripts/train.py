from src.utils.config import load_config
from src.data.dataset import get_dataloaders
from src.models.embedding_model import EmbeddingModel
from src.models.loss import MatryoshkaInfoNCELoss
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from dotenv import load_dotenv
import argparse
import wandb

load_dotenv()

logger = get_logger(__file__)

parser = argparse.ArgumentParser(description='Train the Matryoshka embedding model')
parser.add_argument('--config', type=str, default=None, help='path to config file')
parser.add_argument('--experiment-name', type=str, required=True, help='name for this experiment run')

args = parser.parse_args()
config = load_config(args.config)

#initializing wandb
wandb.init(project='matryoshka-elastic-embeddings', name=args.experiment_name, config=config)

#setting up the data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=config['training']['batch_size'])

#setting up the model and loss
model = EmbeddingModel(model_name=config['model']['name'])

loss_fn = MatryoshkaInfoNCELoss(temperature=config['training']['temperature'], dims=config['matryoshka']['dims'])

logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

#training
trainer = Trainer(model, loss_fn, train_loader, val_loader, config, experiment_name=args.experiment_name)
trainer.train()

wandb.finish()
logger.info('Finished training')
