import datasets
from renumics import spotlight
import argparse

parser = argparse.ArgumentParser(description="View dataset")
parser.add_argument("--name", type=int, default=0, help="dataset name")
args = parser.parse_args()

ds = datasets.load_from_disk(args.name)

spotlight.show(ds)