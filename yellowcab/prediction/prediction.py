#%%
from sklearn.model_selection import train_test_split
from yellowcab.io.input import read_parquet_dataset,read_parquet, read_geo_dataset
from yellowcab.cleaning import clean_dataset
from yellowcab.io.output import write_parquet
from yellowcab.io.utils import get_data_path
from yellowcab.preprocessing import transform_columns
from yellowcab.io.output import save_model
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import calendar
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
from sklearn import metrics
import pandas as pd

#%%
def _get_random_state():
    """
    This function defines our random state, so that it's the same in
    every application
    ----------------------------------------------
    :return:
        int: the integer used for random state paramenter
    """
    RANDOM_STATE = 7
    return RANDOM_STATE

def _get_column_description_for_prediction():
    """
    This function defines the categories our features belong to.
    ----------------------------------------------
    :return:
        dictionary: Dictionary containing our feature categories
                    with the associated attributes.
    """
    #why not all features listed?
    column_description = {'cyclical_features':['pickup_month','pickup_day','pickup_hour',
                                               'dropoff_hour','dropoff_day','dropoff_month'],
                      'categorical_features':['RatecodeID', 'payment_type'],
                      'temporal_features':['pickup_datetime','dropoff_datetime'],
                      'spatial_features':['PULocationID','DOLocationID','centers_lat_dropoff',
                                          'centers_long_dropoff','centers_lat_pickup','centers_long_pickup']}
    return column_description

def _make_data_preparation(df, prediction_type, target):
    """
    TODO: As soon as Simon has corrected weekend in cleaning, line can be removed.
    This function reduces the dataframe to one containing only relevant features
    for prediction purposes.
    ----------------------------------------------
    :param df (pandas.DataFrame): the given pandas data frame with all initial features
           prediction_type (String): Denotes whether used for regression or classification.
           :param target: Dependent variable for prediction purposes.
    :return: pandas.DataFrame: data frame containing only those features which
             are relevant for prediction.

    """
    if prediction_type == "regression":
        if target == "trip_distance":
            column_description = _get_column_description_for_prediction()
            new_df = transform_columns(df,column_description) #for what transform columns?
            new_df.drop(column_description.get('spatial_features'), inplace=True, axis=1)
            new_df.drop(column_description.get('temporal_features'),inplace=True, axis=1)
            new_df.pop('weekend') # remove after Simons correction
        if target == "fare_amount":
            column_description = _get_column_description_for_prediction()
            new_df = transform_columns(df, column_description)
            column_description["categorical_features"] = new_df.loc[:, new_df.columns.str.contains(
                '^Rate')].columns.tolist(), new_df.loc[:, new_df.columns.str.contains('^pay')].columns.tolist()
            column_description["categorical_features"] = sum(column_description["categorical_features"], [])
            new_df.drop(column_description.get('categorical_features'), inplace=True, axis=1)
            new_df.drop(column_description.get('temporal_features'), inplace=True, axis=1)

    else:
        column_description = _get_column_description_for_prediction()
        column_description['categorical_features'] = ['RatecodeID'] #payment type should not be transofrmed here
        new_df = transform_columns(df,column_description) #for what transform columns?
        new_df.drop(column_description.get('spatial_features'),inplace=True, axis=1)
        new_df.drop(column_description.get('temporal_features'),inplace=True, axis=1)
        new_df.pop('weekend') # remove after Simons correction

    #if prediction_type == "regression":
    #    df = pd.get_dummies(df)
    return new_df

def _make_pipeline(model,scaler_type, numerical_features, model_name, use_sampler=False, sampler=None):
    """
    This function assembles several steps that can be cross-validated together
    while setting different parameters.
    ----------------------------------------------
    :param prediction_type (String):
        Denotes whether used for regression or classification.
    :param use_sampler: For testing whether the use of over-/ under-sampling improves our metrics.
    :param model: Used model for prediction.
    :return: Pipeline: Sequentially applies the list of transforms
             and a final estimator.
    """
    steps = []
    preprocessing_steps = []

    if scaler_type == 'Robust':
        scaler = ("robust_scaler", RobustScaler()) #with outlier detection on top
        preprocessing_steps.append(scaler)
    else:
        scaler = ("standard_scaler", StandardScaler())
        preprocessing_steps.append(scaler)

    preprocessor_pipeline= Pipeline(steps=preprocessing_steps)
    preprocessor = ColumnTransformer(
        transformers=[('num', preprocessor_pipeline, numerical_features)])
    preprocessor = ("preprocessor", preprocessor_pipeline)
    steps.append(preprocessor)
    if use_sampler:
        steps.append(sampler)
    prediction_model=(model_name,model)
    steps.append(prediction_model)
    return Pipeline(steps=steps)

def _make_train_test_split(df,target):
    """
    This function splits the input data set into a train an a test set, each
    for the regressors X and the dependent variable y.
    ----------------------------------------------
    :param df (pandas.DataFrame): the given pandas data frame containing data which
                                  need to be split into train and test data sets.
    :param target: Dependent variable for prediction purposes.
    :return:
        X_train (pandas.DataFrame): Regressors used for training a model
        X_test (pandas.DataFrame): Regressors used for testing a model
        y_train (pandas.Series): Target values for training a model
        y_test (pandas.Series): Target values for testing a model
    """
    y = df.pop(target)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_get_random_state())
    return X_train, X_test, y_train, y_test

def _print_prediction_scores(prediction_type, y_test, X_test, pipeline):
    """
    This function prints the prediction scores for either a regression or
    a classification.
    ----------------------------------------------
    :param prediction_type (String): Denotes whether used for regression or classification.
    :param y_test (pandas.Series): Target values for testing a model
    :param X_test (pandas.DataFrame): Regressors used for testing a model
    :param pipeline: Sequentially applies a list of transforms and a final estimator.
    :return:
    """
    if prediction_type == "classification":
        print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
    else:
        print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, pipeline.predict(X_test))))
        print("MAE: ", metrics.mean_absolute_error(y_test, pipeline.predict(X_test)))
        print("R2: ", metrics.r2_score(y_test, pipeline.predict(X_test)))



def make_predictions(df, prediction_type, target, model, model_name, scaler_type, use_sampler=False,
                     sampler=None):
        """
        This function predicts and prints the prediction scores of a prediction
        task dynamically defined by the input parameters.
        ----------------------------------------------
        :param df (pandas.DataFrame): the given pandas data frame containing data
                                      used for prediction.
        :param prediction_type (String): Denotes whether used for regression or classification.
        :param target: Dependent variable for prediction purposes.
        :return:
        """
        df = _make_data_preparation(df, prediction_type, target = target)
        numerical_features = df.select_dtypes(include=['float64']).columns
        X_train, X_test, y_train, y_test = _make_train_test_split(df=df, target=target)
        pipeline = _make_pipeline(model = model, model_name = model_name, scaler_type =scaler_type,
                                  numerical_features= numerical_features,
                                  use_sampler=use_sampler, sampler=sampler) #parameter overloading
        if use_sampler:
            name = sampler
            X_train, y_train = name.fit_sample(X_train, y_train)
        pipeline = pipeline.fit(X_train, y_train)
        save_model(model)
        _print_prediction_scores(prediction_type=prediction_type, y_test=y_test, X_test=X_test,
                                 pipeline=pipeline)


def make_baseline_predictions(df):
    """
    This function performs baseline predictions for both classification and
    regression. Those can be used to evaluate fall other machine learning algorithms,
    as it provides the required point of comparison.
    ----------------------------------------------
    :param df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
    :return:
    """
    classification_model = LogisticRegression(multi_class="multinomial", class_weight="balanced",
                                              n_jobs=-1, random_state=_get_random_state(), max_iter=5000)
    regression_model = LinearRegression()

    # classification for "payment_type"
    make_predictions(
        df=df, prediction_type="classification", target ="payment_type", model = classification_model,
        scaler_type= "standard_scaler", model_name="Classification"
    )

    # classification for "payment_type" using under-sampling (near-miss)


    # regression for "trip_distance"
    make_predictions(
        df=df, prediction_type="regression", target ="trip_distance", model = regression_model,
        scaler_type= "standard_scaler", model_name="Regression 1"
    )

    # regression for "fare_amount"
    make_predictions(
        df=df, prediction_type="regression", target="fare_amount", model=regression_model,
        scaler_type="standard_scaler", model_name="Regression 2"
    )


# NEAR MISS anstatt classbased balane done but not wirking
# save to pickle für alle modelle done
# prediction documented done
# phillips functions documented

#%%


