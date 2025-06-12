import click
from game import Game
from pathlib import Path

@click.group()
def cli():
    """Autonomous Driver CLI"""
    pass

@cli.command()
@click.option('--model-path', type=Path, help='Path to the trained model')
def run(model_path: Path):
    """Run the autonomous driver with a trained model"""
    game = Game()
    game.run()


@cli.command()
@click.option('--epochs', type=int, default=100, help='Number of training epochs')
@click.option('--batch-size', type=int, default=32, help='Training batch size')
@click.option('--learning-rate', type=float, default=0.001, help='Learning rate')
def train(epochs, batch_size, learning_rate):
    """Train the autonomous driver model"""
    click.echo(f"Training model for {epochs} epochs")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Learning rate: {learning_rate}")
    # TODO: Implement the actual training logic

if __name__ == '__main__':
    cli()
