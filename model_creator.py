import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def remove_or_replace_nan_values(df):
    df["MasVnrType"] = df["MasVnrType"].dropna
    df["MasVnrArea"] = df["MasVnrArea"].dropna
    df["BsmtQual"] = df["BsmtQual"].fillna("No")
    df["BsmtCond"] = df["BsmtCond"].fillna("No")
    df["BsmtExposure"] = df["BsmtExposure"].dropna
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna("No")
    df["BsmtFinType2"] = df["BsmtFinType2"].fillna("No")
    df["Electrical"] = df["Electrical"].dropna
    df["FireplaceQu"] = df["FireplaceQu"].fillna("No")
    df["GarageType"] = df["GarageType"].fillna("No")
    df["GarageQual"] = df["GarageQual"].fillna("No")
    df["GarageCond"] = df["GarageCond"].fillna("No")
    df["PoolQC"] = df["PoolQC"].fillna("No")
    df["Fence"] = df["Fence"].fillna("No")
    df["MiscFeature"] = df["MiscFeature"].fillna("No")
    df["Alley"] = df["Alley"].fillna("No")
    df["GarageYrBlt"] = df["GarageYrBlt"].dropna
    df["GarageFinish"] = df["GarageFinish"].dropna
    df["LotFrontage"] = df["LotFrontage"].dropna
    return df


def choose_columns_for_estimation(df_train_dummies):
    columns_for_estimation = []
    for column_name, column_corr in df_train_dummies.corr()["SalePrice"].items():
        if (column_corr > 0.6 or column_corr < -0.6) and column_corr != 1:
            columns_for_estimation.append(column_name)
    return columns_for_estimation


def remove_outliers_in_columns(columns_for_estimation, df_train_dummies):
    for column in columns_for_estimation:
        Q1 = np.percentile(df_train_dummies[column], 25, method="midpoint")

        Q3 = np.percentile(df_train_dummies[column], 75, method="midpoint")
        IQR = Q3 - Q1
        upper = np.where(df_train_dummies[column] >= (Q3 + 1.5 * IQR))
        lower = np.where(df_train_dummies[column] <= (Q1 - 1.5 * IQR))
        for item in upper[0]:
            try:
                df_train_dummies.drop(item, inplace=True)
            except:
                continue
        for item in lower[0]:
            try:
                df_train_dummies.drop(item, inplace=True)
            except:
                continue
        return df_train_dummies


def change_area_format(df):
    df["TotalBsmtSF"] = df["TotalBsmtSF"] * 0.09290304
    df["1stFlrSF"] = df["1stFlrSF"] * 0.09290304
    df["GrLivArea"] = df["GrLivArea"] * 0.09290304
    df["GarageArea"] = df["GarageArea"] * 0.09290304
    return df


def model_to_estimate_player_value(dataframe_train, columns_list):
    y = dataframe_train["SalePrice"]
    X = dataframe_train[columns_list]
    scaler_train = MinMaxScaler()
    scaler_train.fit(X)
    scaler_train.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y)

    rf = RandomForestRegressor()
    model = rf.fit(X_train, y_train.values)
    model_score = rf.score(X_test, y_test.values)
    return (model, model_score, scaler_train)


def generate_best_model(columns_for_estimation, df_train_dummies, n=20):
    best_model = None
    best_score = 0
    best_scaler = None
    for run in range(1, n):
        model, model_score, scaler = model_to_estimate_player_value(
            df_train_dummies, columns_for_estimation
        )
        if model_score > best_score:
            best_score = model_score
            best_model = model
            best_scaler = scaler
    return best_model, best_scaler, best_score


if __name__ == "__main__":
    df_train_raw = pd.read_csv("train.csv", index_col="Id")
    df_train = df_train_raw.copy()
    df_train = remove_or_replace_nan_values(df_train)
    df_train_dummies = pd.get_dummies(df_train)
    df_train_dummies = change_area_format(df_train_dummies)
    columns_for_estimation = choose_columns_for_estimation(df_train_dummies)
    df_train_dummies = remove_outliers_in_columns(
        columns_for_estimation, df_train_dummies
    )
    model, model_score, scaler_train = model_to_estimate_player_value(
        df_train_dummies, columns_for_estimation
    )
    model, scaler, score = generate_best_model(
        columns_for_estimation, df_train_dummies, n=20
    )
    print(f"Model score: {round(score, 2)}")
    joblib.dump(model, "model.pkl")
