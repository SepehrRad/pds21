import math

import xgboost as xgb
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import NearMiss
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from yellowcab.feature_engineering import add_relevant_features
from yellowcab.io.output import save_model
from yellowcab.io.utils import flatten_list, get_zone_information
from yellowcab.preprocessing import transform_columns


def _get_random_state():
    """
    This function defines our random state, so that it's the same in
    every application
    ----------------------------------------------
    :return:
        int: the integer used for random state parameter
    """
    RANDOM_STATE = 7
    return RANDOM_STATE


def _make_data_preparation(df, relevant_features):
    """
    This function reduces the dataframe to one containing only relevant features
    for prediction purposes.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame with all initial features
           relevant_features (list):
    :return: pandas.DataFrame: Data frame containing only those features which
             are relevant for prediction.
    """
    df = get_zone_information(df, zone_file="taxi_zones.csv")
    mask = flatten_list(list(relevant_features.values()))
    df = df[mask]
    df = transform_columns(df=df, col_dict=relevant_features, drop_first=True)

    return df


def _make_pipeline(
    model, model_name, feature_selector=None, feature_selection=False, scaler_type=None
):
    """
    This function assembles several steps that can be cross-validated together
    while setting different parameters.
    ----------------------------------------------
    :param model: Used model for prediction.
           scaler_type (boolean): What scaler should be used to transform our data.
           model_name (String): Name of used model for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
           feature_selector: What feature selector should be applied.
    :return: sklearn.pipeline: Sequentially applies the list of transforms
             and a final estimator.
    """
    steps = []
    if scaler_type is not None:
        if scaler_type == "Robust":
            scaler = ("robust_scaler", RobustScaler())  # with outlier detection on top
            steps.append(scaler)
        else:
            scaler = ("standard_scaler", StandardScaler())
            steps.append(scaler)
    if feature_selection and feature_selector is not None:
        feature_selector_step = ("feature_selector", feature_selector)
        steps.append(feature_selector_step)
    prediction_model = (model_name, model)
    steps.append(prediction_model)
    return Pipeline(steps=steps)


def _make_train_test_split(df, target, use_sampler, sampler):
    """
    This function splits the input data set into a train an a test set, each
    for the regressors X and the dependent variable y.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame containing data which
                                  need to be split into train and test data sets.
           target (String): Dependent variable for prediction purposes.
           use_sampler (boolean): Denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: What sampler should be used for over-/ under-sampling.
    :return:
        X_train (pandas.DataFrame): Regressors used for training a model
        X_test (pandas.DataFrame): Regressors used for testing a model
        y_train (pandas.Series): Target values for training a model
        y_test (pandas.Series): Target values for testing a model
    """
    y = df.pop(target)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=_get_random_state()
    )
    if use_sampler:
        test = sampler
        X_train, y_train = test.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def _print_prediction_scores(prediction_type, y_test, X_test, pipeline):
    """
    This function prints the prediction scores for either a regression or
    a classification.
    ----------------------------------------------
    :param prediction_type (String): Denotes whether used for regression or classification.
           y_test (pandas.Series): Target values for testing a model
           X_test (pandas.DataFrame): Regressors used for testing a model
           pipeline (sklearn.pipeline): Sequentially applies a list of transforms and a final estimator.
    :return:
    """
    if prediction_type == "classification":
        print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
    else:
        y_pred = pipeline.predict(X_test)
        print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred): .3f}")
        print(f"MSE: {metrics.mean_squared_error(y_test, y_pred): .3f}")
        print(f"RMSE: {math.sqrt(metrics.mean_squared_error(y_test, y_pred)): .3f}")
        print(f"R2: {100 * metrics.r2_score(y_test, y_pred): .3f} %")


def _get_information_for_feature_selection(pipeline, X_train):
    """

    :param pipeline:
           X_train:
    :return:
    """
    selected_feature_mask = pipeline.named_steps["feature_selector"].get_support()
    original_features = len(X_train.columns)
    selected_features = len(X_train.columns[selected_feature_mask])
    print(
        f"{selected_features} features were selected for prediction from the original {original_features} features."
    )


def make_predictions(
    df,
    prediction_type,
    target,
    model,
    model_name,
    scaler_type,
    relevant_features,
    feature_selector=None,
    feature_selection=False,
    use_sampler=False,
    sampler=None,
):
    """
    This function predicts and prints the prediction scores of a prediction
    task dynamically defined by the input parameters.
    ----------------------------------------------
    :param df (pandas.DataFrame): the given pandas data frame containing data used for prediction.
           prediction_type (String: Denotes whether used for regression or classification.
           target (String): Dependent variable for prediction purposes.
           model: Used model for prediction.
           model_name (String): Name of used model for prediction.
           scaler_type (String): scaler_type: What scaler should be used to transform our data.
           relevant_features (list): Those features that should be used for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
           use_sampler (boolean): denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: what sampler should be used for over-/ under-sampling.
    :return:
    """
    print(
        f"\nPredicted target: {target}, model name: {model_name}, prediction type: {prediction_type}"
    )
    df = _make_data_preparation(df, relevant_features=relevant_features)
    X_train, X_test, y_train, y_test = _make_train_test_split(
        df=df, target=target, sampler=sampler, use_sampler=use_sampler
    )
    if feature_selection:
        pipeline = _make_pipeline(
            model=model,
            model_name=model_name,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            feature_selector=feature_selector,
        )
    else:
        pipeline = _make_pipeline(
            model=model, model_name=model_name, scaler_type=scaler_type
        )
    pipeline = pipeline.fit(X_train, y_train)
    if feature_selection:
        _get_information_for_feature_selection(pipeline=pipeline, X_train=X_train)
    save_model(pipeline.named_steps[model_name], model_name)
    _print_prediction_scores(
        prediction_type=prediction_type, y_test=y_test, X_test=X_test, pipeline=pipeline
    )


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
    classification_model = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        random_state=_get_random_state(),
        max_iter=5000,
    )
    regression_model = LinearRegression(n_jobs=-1)

    nr = NearMiss()

    # classification for "payment_type"
    make_predictions(
        df=df,
        prediction_type="classification",
        target="payment_type",
        model=classification_model,
        relevant_features={
            "target": "payment_type",
            "cyclical_features": [
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "dropoff_hour",
                "dropoff_day",
                "dropoff_month",
            ],
            "categorical_features": ["Zone_pickup", "Zone_dropoff"],
            "numerical_features": [
                "passenger_count",
                "trip_distance",
                "total_amount",
                "trip_duration_minutes",
            ],
        },
        scaler_type="standard_scaler",
        model_name="base_clas_payment_type",
    )

    # classification for "payment_type" using under-sampling (near-miss)
    make_predictions(
        df=df,
        prediction_type="classification",
        target="payment_type",
        model=classification_model,
        relevant_features={
            "target": "payment_type",
            "cyclical_features": [
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "dropoff_hour",
                "dropoff_day",
                "dropoff_month",
            ],
            "categorical_features": ["Zone_pickup", "Zone_dropoff"],
            "numerical_features": [
                "passenger_count",
                "trip_distance",
                "total_amount",
                "trip_duration_minutes",
            ],
        },
        scaler_type="standard_scaler",
        model_name="base_clas_payment_type_nm",
        use_sampler=True,
        sampler=nr,
    )

    # regression for "trip_distance"
    make_predictions(
        df=df,
        prediction_type="regression",
        target="trip_distance",
        model=regression_model,
        relevant_features={
            "target": "trip_distance",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
            "numerical_features": ["passenger_count"],
        },
        scaler_type=None,
        model_name="base_reg_trip_distance",
    )

    # regression for "fare_amount"
    make_predictions(
        df=df,
        prediction_type="regression",
        target="fare_amount",
        model=regression_model,
        scaler_type=None,
        relevant_features={
            "target": "fare_amount",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
            "numerical_features": ["passenger_count"],
        },
        model_name="base_reg_fare_amount",
        use_sampler=False,
        sampler=None,
    )


def trip_distance_regression(df, feature_selection=False):
    """
    This function predicts the trip distance with using xgboost. Information is used that was available at
    the start of the trip (including additional created features).
    :param df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
    :return:
    """
    feature_selector = SelectFromModel(Lasso(alpha=0.1))
    df = add_relevant_features(df, "pickup_datetime")
    if feature_selection:
        make_predictions(
            df=df,
            prediction_type="regression",
            target="trip_distance",
            relevant_features={
                "target": "trip_distance",
                "categorical_features": ["Zone_dropoff", "Zone_pickup"],
                "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
                "numerical_features": [
                    "passenger_count",
                    "Holiday",
                    "covid_lockdown",
                    "covid_school_restrictions",
                    "covid_new_cases",
                ],
            },
            feature_selector=feature_selector,
            feature_selection=True,
            model=xgb.XGBRegressor(n_estimators=100),
            model_name="xg_boost",
            scaler_type=None,
            use_sampler=False,
            sampler=None,
        )
    else:
        make_predictions(
            df=df,
            prediction_type="regression",
            target="trip_distance",
            relevant_features={
                "target": "trip_distance",
                "categorical_features": ["Zone_dropoff", "Zone_pickup"],
                "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
                "numerical_features": [
                    "passenger_count",
                    "Holiday",
                    "covid_lockdown",
                    "covid_school_restrictions",
                    "covid_new_cases",
                ],
            },
            model=xgb.XGBRegressor(n_estimators=100),
            model_name="xg_boost",
            scaler_type=None,
            use_sampler=False,
            sampler=None,
        )
