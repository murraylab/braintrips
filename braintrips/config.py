from pathlib import Path
from os import path

root_dir = Path(path.dirname(path.dirname(
    path.abspath(path.join(__file__)))))    # package root
data_dir = root_dir.joinpath("data")  # data directory
balsa_dir = data_dir.joinpath("balsa")  # includes data provided on BALSA
