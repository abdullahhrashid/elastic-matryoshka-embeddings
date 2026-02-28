from transformers import get_linear_schedule_with_warmup
from src.utils.logger import get_logger
from tqdm import tqdm
import torch
import wandb
import os

logger = get_logger(__file__)

class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, config, experiment_name):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.warmup_ratio = config['training']['warmup_ratio']
        self.checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], experiment_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f'Using device: {self.device}')

        #setting up the optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logger.info(f'Total steps: {total_steps}, warmup steps: {warmup_steps}')

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            logger.info(f'Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.scheduler.get_last_lr()[0],
            })

            self.save_checkpoint(epoch)

        logger.info('Training complete')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', leave=True)

        for step, batch in enumerate(progress_bar):
            #moving all tensors to the device
            anchor = {k: v.to(self.device) for k, v in batch['anchor'].items()}
            positive = {k: v.to(self.device) for k, v in batch['positive'].items()}
            negative = {k: v.to(self.device) for k, v in batch['negative'].items()}

            #forward pass through the model for all three texts
            anchor_embs = self.model(anchor)
            positive_embs = self.model(positive)
            negative_embs = self.model(negative)

            loss = self.loss_fn(anchor_embs, positive_embs, negative_embs)

            #backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_postfix(loss=f'{avg_loss:.4f}', lr=f'{lr:.2e}')

            if (step + 1) % 100 == 0:
                wandb.log({
                    'step_train_loss': avg_loss,
                    'step': epoch * len(self.train_loader) + step,
                    'step_lr': lr,
                })

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]', leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                anchor = {k: v.to(self.device) for k, v in batch['anchor'].items()}
                positive = {k: v.to(self.device) for k, v in batch['positive'].items()}
                negative = {k: v.to(self.device) for k, v in batch['negative'].items()}

                anchor_embs = self.model(anchor)
                positive_embs = self.model(positive)
                negative_embs = self.model(negative)

                loss = self.loss_fn(anchor_embs, positive_embs, negative_embs)
                total_loss += loss.item()

                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix(val_loss=f'{avg_loss:.4f}')

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)

        #uploading to wandb as a versioned artifact
        artifact = wandb.Artifact(
            name=f'checkpoint-epoch-{epoch+1}',
            type='model',
            description=f'MRL checkpoint after epoch {epoch+1}'
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        logger.info(f'Saved checkpoint to {checkpoint_path} and uploaded to WandB')
