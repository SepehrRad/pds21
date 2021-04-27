import math

import numpy as np
import pandas as pd


def _cyclical_feature_transformer(cyclical_col):

    """
    This function maps cyclical features to two distinct components (sine & cosine) & return them as vector.
    ----------------------------------------------
    :param
        cyclical_col (numpy.ndarray): cyclical feature vector

    :returns
        numpy.ndarray: sine & cosine vectors for the given feature vector
        {
        sine,
        cosine
        }
    """
    maximum_value = np.amax(cyclical_col)
    sin_comp = np.sin((2 * math.pi * cyclical_col) / maximum_value)
    cos_comp = np.cos((2 * math.pi * cyclical_col) / maximum_value)
    return sin_comp, cos_comp


def _categorical_feature_transformer(df, categorical_col_names, drop_first=False):

    """
    This function encodes the categorical variables in a given data frame using one hot encoding method.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given pandas data frame with categorical features
        categorical_col_names (string[]): the name of the categorical features as a list
        drop_first (bool): the decision to drop one category per feature

    :returns
        pandas.Series: the encoded categorical features
    """
    df[categorical_col_names] = df[categorical_col_names].astype("str")
    return pd.get_dummies(df[categorical_col_names], drop_first=drop_first)


def transform_columns(df, col_dict, drop_cols=True, drop_first=False):

    """
    This function transforms the cyclical & categorical columns of any given data frame.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given pandas data frame with categorical & cyclical features
        col_dict (dict): the name of the categorical & cyclical features
        drop_first (bool): the decision to drop one category for categorical features
        drop_cols(bool): the decision to remove the original features from the given data frame after transformation

    :returns
        pandas.DataFrame: the data frame with transformed features

    :raises
        ValueError: if col_dict or df are empty/null
    """

    if col_dict is None or len(col_dict) == 0:
        raise ValueError("The columns dictionary can not be null!")
    if df is None:
        raise ValueError("The data frame can not be null!")

    if len(col_dict.get("cyclical_features")) != 0:
        for feature in col_dict.get("cyclical_features"):
            (
                df[f"{feature}_sine"],
                df[f"{feature}_cosine"],
            ) = _cyclical_feature_transformer(df[feature])

    if len(col_dict.get("categorical_features")) != 0:
        _ = _categorical_feature_transformer(
            df, col_dict.get("categorical_features"), drop_first=drop_first
        )
        df = df.join(_)

    if drop_cols:
        df.drop(col_dict.get("cyclical_features"), inplace=True, axis=1)
        df.drop(col_dict.get("categorical_features"), inplace=True, axis=1)

    return df
