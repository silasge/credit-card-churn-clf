import argparse
import sys

from loguru import logger

from credit_card_churn_clf.models.modeling_funcs import train_model


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--export_pkl_to", type=str)
    parser.add_argument("--print_log", action="store_true")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    train_model(
        data=args.data,
        model=args.model,
        cv=args.cv,
        scoring=args.scoring,
        n_iter=args.n_iter,
        random_state=args.random_state,
        export_pkl_to=args.export_pkl_to,
        print_log=args.print_log,
    )


if __name__ == "__main__":
    main()
