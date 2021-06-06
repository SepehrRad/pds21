import xgboost as xgb
from imblearn.under_sampling import NearMiss
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from .. import io
from ..feature_engineering import add_relevant_features
from ..io import get_random_state
from ..prediction import make_predictions


def make_baseline_predictions(df):
    """
    This function performs baseline predictions for both classification and
    regression. Those can be used to evaluate fall other machine learning algorithms,
    as it provides the required point of comparison.
    ----------------------------------------------
    :param
            df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
    :return:
    """
    classification_model = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        random_state=get_random_state(),
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


# hyperparameter search
# finale version nur mit lasso ausgewählt
# distazen


def trip_distance_regression_base(df):
    """
    This function predicts the trip distance with using xgboost. Information is used that was available at
    the start of the trip (including additional created features).
    :param
            df (pandas.DataFrame): the given pandas data frame containing data
                                  used for prediction.
           feature_selection (boolean): If features should be selected to improve estimators’
                                        accuracy scores or to boost their performance.
    """

    feature_selector = SelectFromModel(Lasso(alpha=0.01))
    df = add_relevant_features(df, "pickup_datetime")

    make_predictions(
        df=df,
        prediction_type="regression",
        target="trip_distance",
        relevant_features={
            "target": "trip_distance",
            "categorical_features": ["Zone_dropoff", "Zone_pickup"],
            "cyclical_features": [],
            "numerical_features": [
                "passenger_count",
                "Holiday",
                "covid_lockdown",
                "covid_school_restrictions",
                "covid_new_cases",
                "pickup_month",
                "pickup_day",
                "pickup_hour",
                "haversine_distance",
                "bearing_distance",
                "manhattan_distance",
                "weekend",
                "weekday",
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
