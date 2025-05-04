import argparse
import os
from pathlib import Path




# Arguments to be passed on the command line when executing the scripts.
def parse_args_KWS() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyword Spotting")
    parser.add_argument(
        "--images",
        type=Path,
        default="KWS/images",
        help=(
            "Path to the images fold  from the George Washington Database "
            "[Default: ./KWS/images]"
        ),
    )
    parser.add_argument(
        "--locations",
        type=Path,
        default="KWS/locations",
        help=(
            "Path to the svgs fold  from the George Washington Database "
            "[Default: ./KWS/locations]"
        ),
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default="KWS/keywords.tsv",
        help=(
            "Path to the tsv file of the keywords from the George Washington Database "
            "[Default: ./KWS/keywords.tsv]"
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default="KWS/train.tsv",
        help=(
            "Path to the tsv file of the training set from the George Washington Database "
            "[Default: ./KWS/train.tsv]"
        ),
    )
    parser.add_argument(
        "--trainscription",
        type=Path,
        default="KWS/transcription.tsv",
        help=(
            "Path to the tsv file of the trainscription set from the George Washington Database "
            "[Default: ./KWS/transcription.tsv]"
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default="KWS/validation.tsv",
        help=(
            "Path to the CSV file of the validation set from the George Washington Database "
            "[Default: ./KWS/validation.tsv]"
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        default="output",
        help="Path to the direcotry where all outputs are saved [Default: ./output/]",
    )
    return parser.parse_args()
