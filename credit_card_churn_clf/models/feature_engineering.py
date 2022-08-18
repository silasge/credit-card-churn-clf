import category_encoders as ce
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer

ord_map = [
    {
        "col": "Education_Level",
        "mapping": {
            "Unknown": -1,
            "Uneducated": 0,
            "High School": 1,
            "College": 2,
            "Graduate": 3,
            "Doctorate": 3,
            "Post-Graduate": 3,
        },
    },
    {
        "col": "Income_Category",
        "mapping": {
            "Unknown": -1,
            "Less than $40K": 0,
            "$40K - $60K": 1,
            "$60K - $80K": 2,
            "$80K - $120K": 3,
            "$120K +": 4,
        },
    },
]

col_transf = ColumnTransformer(
    [
        ("gender_marital_oh", OneHotEncoder(), ["Gender", "Marital_Status"]),
        ("card_type_oh", OneHotEncoder(min_frequency=0.1), ["Card_Category"]),
        (
            "edu_income_ord",
            ce.OrdinalEncoder(mapping=ord_map),
            ["Education_Level", "Income_Category"],
        ),
        (
            "log_trans",
            FunctionTransformer(func=np.log1p),
            [
                "Credit_Limit",
                "Total_Revolving_Bal",
                "Total_Trans_Amt",
                "Avg_Utilization_Ratio",
            ],
        ),
        (
            "std_scaler",
            StandardScaler(),
            [
                "Customer_Age",
                "Months_on_book",
                "Total_Amt_Chng_Q4_Q1",
                "Total_Trans_Ct",
                "Total_Ct_Chng_Q4_Q1",
            ],
        ),
    ],
    remainder="passthrough",
)
