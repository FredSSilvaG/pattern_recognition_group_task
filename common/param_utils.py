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
        "--kwsTest",
        type=Path,
        default="KWS-test/",
        help=(
            "Path to the test images fold  from the George Washington Database "
            "[Default: ./KWS-test]"
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


def parse_args_SV() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signature Verification")
    parser.add_argument(
        "--base",
        type=Path,
        default="SignatureVerification",
        help=(
            "Path to the base fold from the MCYT Signatures base"
            "[Default: ./SignatureVerification-test]"
        ),
    )
    parser.add_argument(
        "--bese_test",
        type=Path,
        default="SignatureVerification-test",
        help=(
            "Path to the base fold from the MCYT Signatures base test"
            "[Default: ./SignatureVerification-test]"
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help=(
            "Top k distance as genuine signature"
            "[Default: 20]"
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_path",
        type=Path,
        default="output/taskThree/report.md",
        help="Path to the direcotry where all outputs are saved [Default: ./output/taskThree/report.md]",
    )
    parser.add_argument(
        "--out-test",
        dest="out_test_path",
        type=Path,
        default="taskThree/test.tsv",
        help="Path to the direcotry where all outputs are saved [Default: ./taskThree/test.tsv]",
    )
    return parser.parse_args()

     