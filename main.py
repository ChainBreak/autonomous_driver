import click
from game import Game
from pathlib import Path
from lit_module import LitModule
import lightning as L
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

@click.group()
def cli():
    """Autonomous Driver CLI"""
    pass

@cli.command()
@click.option('--checkpoint-path', type=Path, help='Path to the checkpoint file')
def run(checkpoint_path: Path):
    """Run the autonomous driver with a trained model"""
    game = Game(checkpoint_path)
    game.setup()
    game.run()


@cli.command()
@click.option('--config-path', type=Path, required=True, help='Path to the config file')
@click.option('--checkpoint-path', type=Path, help='Path to the checkpoint file')
def train(config_path: Path, checkpoint_path: Path):
    """Train the autonomous driver model"""

    config = load_config(config_path)

    model = LitModule(config)

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="autonomous-driver",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # metric to monitor
        dirpath=logger.log_dir + '/checkpoints',
        filename='autonomous-driver-{step:06d}-{train_loss:.2f}',
        save_top_k=5,  # save top 5 checkpoints
        mode='min',  # minimize the monitored metric
        save_last=True,  # also save the last checkpoint
    )

    trainer = L.Trainer(
        max_epochs=None,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        ckpt_path=checkpoint_path,
    )

def load_config(config: Path):
    with open(config, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
   

if __name__ == '__main__':
    cli()
