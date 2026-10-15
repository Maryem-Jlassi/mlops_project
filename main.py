# main.py
"""
Point d'entrée CLI pour exécuter le pipeline ML.
Usage:
    python main.py prepare
    python main.py train
    python main.py evaluate
    python main.py save
    python main.py load
    python main.py all
"""

import argparse
import sys
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

CSV_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def cmd_prepare(args):
    x_train, x_test, y_train, y_test = prepare_data(CSV_PATH)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")


def cmd_train(args):
    x_train, x_test, y_train, y_test = prepare_data(CSV_PATH)
    model = train_model(x_train, y_train)
    save_model(model, MODEL_PATH)


def cmd_evaluate(args):
    x_train, x_test, y_train, y_test = prepare_data(CSV_PATH)
    model = load_model(MODEL_PATH)
    evaluate_model(model, x_test, y_test)


def cmd_save(args):
    x_train, x_test, y_train, y_test = prepare_data(CSV_PATH)
    model = train_model(x_train, y_train)
    save_model(model, MODEL_PATH)


def cmd_load(args):
    model = load_model(MODEL_PATH)
    print(model)


def cmd_all(args):
    x_train, x_test, y_train, y_test = prepare_data(CSV_PATH)
    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Churn")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("save").set_defaults(func=cmd_save)
    sub.add_parser("load").set_defaults(func=cmd_load)
    sub.add_parser("all").set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
