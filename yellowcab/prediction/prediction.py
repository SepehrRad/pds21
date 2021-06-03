import math

from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import NearMiss
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from yellowcab.io.output import save_model
from yellowcab.io.utils import get_zone_information
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


def _get_column_description_for_prediction():
    """
    This function defines the categories our features belong to.
    ----------------------------------------------
    :return:
        dictionary: Dictionary containing our feature categories with the associated attributes.
    """
    column_description = {
        "cyclical_features": [
            "pickup_month",
            "pickup_day",
            "pickup_hour",
            "dropoff_hour",
            "dropoff_day",
            "dropoff_month",
        ],
        "categorical_features": ["RatecodeID", "payment_type"],
        "temporal_features": ["pickup_datetime", "dropoff_datetime"],
        "spatial_features": [
            "LocationID_pickup",
            "Borough_pickup",
            "Zone_pickup",
            "service_zone_pickup",
            "LocationID_dropoff",
            "Borough_dropoff",
            "Zone_dropoff",
            "service_zone_dropoff",
        ],
    }
    return column_description


def _make_data_preparation(df, prediction_type, target):
    """

    This function reduces the dataframe to one containing only relevant features
    for prediction purposes.
    ----------------------------------------------
    :param df (pandas.DataFrame): The given pandas data frame with all initial features
           prediction_type: Denotes whether used for regression or classification.
           target: Dependent variable for prediction purposes.
    :return: pandas.DataFrame: Data frame containing only those features which
             are relevant for prediction.
    """
    df = get_zone_information(df, zone_file="taxi_zones.csv")
    if prediction_type == "regression":
        print(f"\n{target} regression")
        relevant_features = [
            target,
            "pickup_month",
            "pickup_day",
            "pickup_hour",
            "Zone_dropoff",
            "Zone_pickup",
            "passenger_count",
        ]

        column_description = {
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": ["pickup_month", "pickup_day", "pickup_hour"],
        }

        df = df[relevant_features]
        df = transform_columns(df=df, col_dict=column_description, drop_first=True)

    else:
        print("\nclassification")
        column_description = _get_column_description_for_prediction()
        # As the target is in itself a categorical variable it should be removed from the column description
        column_description_cat = column_description.get("categorical_features")
        column_description_cat.remove(target) if column_description_cat else None

        df = transform_columns(df=df, col_dict=column_description)
        df.drop(column_description.get("spatial_features"), inplace=True, axis=1)
        df.drop(column_description.get("temporal_features"), inplace=True, axis=1)

    return df


def _make_pipeline(model, model_name, scaler_type=None):
    """
    This function assembles several steps that can be cross-validated together
    while setting different parameters.
    ----------------------------------------------
    :param model: Used model for prediction.
           scaler_type: What scaler should be used to transform our data.
           model_name: Name of used model for prediction.
    :return: Pipeline: Sequentially applies the list of transforms
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
           target: Dependent variable for prediction purposes.
           use_sampler: Denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
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
    :param prediction_type: Denotes whether used for regression or classification.
           y_test (pandas.Series): Target values for testing a model
           X_test (pandas.DataFrame): Regressors used for testing a model
           pipeline: Sequentially applies a list of transforms and a final estimator.
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


def make_predictions(
    df,
    prediction_type,
    target,
    model,
    model_name,
    scaler_type,
    use_sampler=False,
    sampler=None,
):
    """
    This function predicts and prints the prediction scores of a prediction
    task dynamically defined by the input parameters.
    ----------------------------------------------
    :param df: the given pandas data frame containing data used for prediction.
           prediction_type: Denotes whether used for regression or classification.
           target: Dependent variable for prediction purposes.
           model: Used model for prediction.
           model_name: Name of used model for prediction.
           scaler_type: scaler_type: What scaler should be used to transform our data.
           sampler_name: Name of the sampler that should be used for over-/ undersampling
           use_sampler: denotes, whether a sampler is used to handle imbalanced data with over-/ under-sampling.
           sampler: what sampler should be used for over-/ under-sampling.
    :return:
    """
    df = _make_data_preparation(df, prediction_type, target=target)
    X_train, X_test, y_train, y_test = _make_train_test_split(
        df=df, target=target, sampler=sampler, use_sampler=use_sampler
    )
    pipeline = _make_pipeline(
        model=model, model_name=model_name, scaler_type=scaler_type
    )
    pipeline = pipeline.fit(X_train, y_train)
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
    :param df: the given pandas data frame containing data
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
        scaler_type="standard_scaler",
        model_name="base_clas_payment_type",
    )

    # classification for "payment_type" using under-sampling (near-miss)
    make_predictions(
        df=df,
        prediction_type="classification",
        target="payment_type",
        model=classification_model,
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
        model_name="base_reg_fare_amount",
        use_sampler=False,
        sampler=None,
    )
