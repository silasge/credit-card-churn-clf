import argparse
import sys

from credit_card_churn_clf.data.data_funcs import (
    import_credit_card_churn_data,
    split_credit_card_churn_data,
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--export_pkl_to", type=str)
    parser.add_argument("--print_log", action="store_true")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    cc_churn_df = import_credit_card_churn_data(args.csv)
    print(args.csv)
    cc_churn_splits = split_credit_card_churn_data(
        cc_churn_df,
        test_size=args.test_size,
        random_state=args.random_state,
        export_pkl_to=args.export_pkl_to,
        print_log=args.print_log,
    )


if __name__ == "__main__":
    main()
