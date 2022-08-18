import argparse
import sys

from credit_card_churn_clf.models.modeling_funcs import (
    get_best_model,
    predict_on_test_set,
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--models_path", nargs="*")
    parser.add_argument("--export_pkl_to", type=str)
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    best_model = get_best_model(args.models_path)
    predict_on_test_set(
        data=args.data,
        best_model=best_model,
        threshold=args.threshold,
        export_pkl_to=args.export_pkl_to,
    )


if __name__ == "__main__":
    main()
