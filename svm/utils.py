import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
import argparse
from keras.preprocessing.image import load_img, img_to_array
from typing import List


def load_images_from_df(df, target_size=(28, 28)):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = 'MNIST-full/' + row['path']
        img = load_img(img_path, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        images.append(img)
        labels.append(row['class'])
    return np.array(images), np.array(labels)

def load_mnist_data():
    train_df = pd.read_csv('MNIST-full/gt-train.tsv', sep='\t', header=None, names=['path', 'class'])
    test_df = pd.read_csv('MNIST-full/gt-test.tsv', sep='\t', header=None, names=['path', 'class'])
    X_train, y_train = load_images_from_df(train_df)
    X_test, y_test = load_images_from_df(test_df)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test


# Reference:  Exercise-1a-Solution
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SVM on MNIST")
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        default="svm/output",
        help="Path to the direcotry where all outputs are saved [Default: ./output/]",
    )
    return parser.parse_args()


TABLE_SEPARATOR_REGEX = re.compile("[^|]")
def create_markdown_table(
    report_result: list[list[float]],
    row_names: list[str],
    col_names: list[str],
    min_width: int = 7,
) -> str:
    first_column_width = max([len(name) for name in row_names])
    column_widths = [max(len(name), min_width) for name in col_names]
    header = [" " * first_column_width] + [
        f"{name:>{width}}" for name, width in zip(col_names, column_widths)
    ]
    rows = [
        [f"{name:<{first_column_width}}"]
        + [
            f"{acc:>{width}}" if idx == 0 else f"{acc:>{width}.4f}"
            for idx, (acc, width) in enumerate(zip(acc_row, column_widths))
        ]
        for name, acc_row in zip(row_names, report_result)
    ]
    header_line = f"| {' | '.join(header)} |"
    separator = TABLE_SEPARATOR_REGEX.sub("-", header_line)
    lines = [header_line, separator] + [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)

def save_markdown_report(
    path: str | os.PathLike,
    report_result: list[list[float]],
    row_names: list[str],
    col_names: list[str],
    best_model: dict,
    best_model_accuracy: str,
    best_model_classification_report:str
):
    table = create_markdown_table(report_result, row_names=row_names, col_names=col_names)
    with open(path, "a", encoding="utf-8") as fd:
        fd.write("\n")
        fd.write("\n")
        fd.write("#SVM on MNIST\n")
        fd.write("\n")
        fd.write(table)
        fd.write("\n")
        fd.write("\n")
        fd.write("Best model's parameters:")
        fd.write(json.dumps(best_model))
        fd.write("\n")
        fd.write("Best SVM accuracy on test set:")
        fd.write(best_model_accuracy)
        fd.write("\n")
        fd.write("Best model classification_report:")
        fd.write("\n")
        fd.write(best_model_classification_report)
        

