import re
from typing import Literal, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from credit_card_churn_clf.utils import export_pkl, add_logger


@add_logger(type="export")
@export_pkl
def import_credit_card_churn_data(
    credit_card_churn_raw: str,
    remove_naive_bayes_cols: bool = True,
    remove_client_id_col: bool = True,
    remove_redundant_col: bool = True,
    export_pkl_to: Optional[str] = None,
    print_log: bool = False,
) -> pd.DataFrame:
    """Import Credit Card Churn dataset

    Args:
        credit_card_churn_raw (str): The file's path
        remove_naive_bayes_cols (bool, optional): Removes naive bayes useless columns.
            Defaults to True.
        remove_client_id_col (bool, optional): Removes client id cols.
            Defaults to True.

    Returns:
        pd.DataFrame: Returns the Credit Card Churn dataset as a Pandas' DataFrame
    """
    cc_churn_df = pd.read_csv(credit_card_churn_raw)
    if remove_naive_bayes_cols:
        pat = re.compile(r"^Naive_Bayes")
        cols_to_keep = [not bool(re.match(pat, col)) for col in cc_churn_df.columns]
        cc_churn_df = cc_churn_df.iloc[:, cols_to_keep]
    if remove_client_id_col:
        cc_churn_df = cc_churn_df.drop("CLIENTNUM", axis=1)
    if remove_redundant_col:
        cc_churn_df = cc_churn_df.drop("Avg_Open_To_Buy", axis=1)
    return cc_churn_df


@add_logger(type="export")
@export_pkl
def split_credit_card_churn_data(
    credit_card_churn_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    export_pkl_to: Optional[str] = None,
    print_log: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cc_churn_train, cc_churn_test = train_test_split(
        credit_card_churn_data, test_size=test_size, random_state=random_state, **kwargs
    )
    return cc_churn_train, cc_churn_test


def read_credit_card_churn_split(
    path: str,
    train_or_test: Literal["train", "test", "all"] = "all",
    split_y: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if train_or_test not in ["train", "test", "all"]:
        raise Exception("Value not accepted for train_or_test")
    splits = joblib.load(path)
    train, test = splits
    if split_y:
        train = _split_credit_card_churn_y(train)
        test = _split_credit_card_churn_y(test)
    if train_or_test == "train":
        return train
    elif train_or_test == "test":
        return test
    elif train_or_test == "all":
        return train, test


def _split_credit_card_churn_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop("Attrition_Flag", axis=1)
    y = label_binarize(
        df[["Attrition_Flag"]], classes=["Existing Customer", "Attrited Customer"]
    ).ravel()
    return X, y
