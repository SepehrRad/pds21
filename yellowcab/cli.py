import click
from . import model


@click.command()
@click.option('--transform/--no-transform', default=False)
@click.option('-i', default=None)
@click.option('-o', default=None)
@click.option('--predict/--no-predict', default=False)
def main(transform, i, o, predict):
    if transform:
        if i is None or o is None:
            print("Insufficient file specification, use -i for input path and -o for output path")
        else:
            print("Transforming data ...")
    elif predict:
        print("Predicting stuff...")


if __name__ == '__main__':
    main()

