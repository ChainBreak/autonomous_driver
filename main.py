import click
from game import Game
from pathlib import Path
from lit_module import LitModule
import lightning as L
import yaml

@click.group()
def cli():
    """Autonomous Driver CLI"""
    pass

@cli.command()
@click.option('--model-path', type=Path, help='Path to the trained model')
def run(model_path: Path):
    """Run the autonomous driver with a trained model"""
    game = Game()
    game.setup()
    game.run()


@cli.command()
@click.option('--config-path', type=Path, required=True, help='Path to the config file')
@click.option('--checkpoint-path', type=Path, help='Path to the checkpoint file')
def train(config_path: Path, checkpoint_path: Path):
    """Train the autonomous driver model"""

    config = load_config(config_path)

    model = LitModule(config)

    trainer = L.Trainer(max_epochs=10)

    trainer.fit(model)

def load_config(config: Path):
    with open(config, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
   

if __name__ == '__main__':
    cli()
