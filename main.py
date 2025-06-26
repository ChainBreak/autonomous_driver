import random
import click
from dataset import RecordedDataset
from history_digest import HistoryDigest
from game import Game
from pathlib import Path
from action_categorizer import ActionCategorizer

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
@click.option('--data-dir', type=Path, help='Path to the data directory')
def train(data_dir: Path):
    """Train the autonomous driver model"""
    click.echo(f"Training model for {data_dir}")
    # TODO: Implement the actual training logic
    history_digest = HistoryDigest.from_window_growth_rate(num_windows=8, growth_rate=2.0)
    action_categorizer = ActionCategorizer(action_vector_length=4)
    dataset = RecordedDataset(data_dir=data_dir, history_digest=history_digest, action_categorizer=action_categorizer)

    for i in range(10):
        item = dataset[random.randint(0, len(dataset)-1)]
        print(item)


@cli.command()
def test():
    target = 0.99
    for i in range(10):
        steps = 2**i
        a = 1 - (1-target)**(1/steps)
        print(f"{steps} {a:.2f}")


    x=1
    y = 0
    y_initial = y
    a = 0.1
    for i in range(10):
        y += a * (x- y )
        y_ = y_initial + (1-(1-a)**(i+1)) * (x-y_initial)
        print(f"{x:.2f} {y:.2f} {y_:.2f}")

if __name__ == '__main__':
    cli()
