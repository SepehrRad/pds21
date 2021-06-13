import os
import typer
from pyfiglet import Figlet
import pickle
import input_output
import cleaning
from yellowcab.feature_engineering import add_relevant_features
from yellowcab.input_output import get_data_path
from yellowcab.prediction.prediction import _make_data_preparation

font = Figlet(font='slant')

# App init
pds_app = typer.Typer()


def _get_relevant_features():
    """
    This function provides the relevant features for each prediction target used two build the optimized models.
    """
    fare_amount_relevant_features = {
        "target": "fare_amount",
        "categorical_features": [],
        "numerical_features": [
            "trip_distance",
            "trip_duration_minutes",
            "pickup_month",
            "pickup_hour",
        ],
        "cyclical_features": [],
    }
    trip_distance_relevant_features = {
        "target": "trip_distance",
        "categorical_features": [],
        "numerical_features": [
            "pickup_month",
            "pickup_day",
            "haversine_distance",
            "bearing_distance",
            "manhattan_distance",
        ],
        "cyclical_features": [],
    }
    payment_type_relevant_features = {
        "target": "payment_type",
        "cyclical_features": [],
        "categorical_features": ["Zone_pickup", "Zone_dropoff"],
        "numerical_features": [
            "covid_lockdown",
            "total_amount",
            "passenger_count",
            "trip_distance",
        ],
        "created_features": [
            "Zone_pickup_JFK Airport",
        ],
    }
    return fare_amount_relevant_features, trip_distance_relevant_features, payment_type_relevant_features


def _load_models(manhattan: bool = False):
    """
    This function loads the pre-trained models into the memory by deserializing the pickle file.
    ----------------------------------------------
    :param
        manhattan(bool): Specifies if the manhattan models should be loaded.
    """
    model_path = os.path.join(get_data_path(), "output")
    if manhattan:
        fare_amount_model = pickle.load(open(f'{model_path}/xgb_model_fare_amount_optimized_manhattan.pkl', "rb"))
        trip_distance_model = pickle.load(open(f'{model_path}/xgb_model_trip_distance_optimized_manhattan.pkl', "rb"))
        payment_type_model = pickle.load(open(f'{model_path}/xgb_model_payment_type_optimized_manhattan.pkl', "rb"))
    else:
        fare_amount_model = pickle.load(open(f'{model_path}/xgb_model_fare_amount_optimized.pkl', "rb"))
        trip_distance_model = pickle.load(
            open(f'{model_path}/xgb_model_trip_distance_optimized.pkl', "rb"))
        payment_type_model = pickle.load(open(f'{model_path}/xgb_model_payment_type_optimized.pkl', "rb"))

    return fare_amount_model, trip_distance_model, payment_type_model


@pds_app.command("transform")
def transform(input_file_path: str, month: int, output_file_name: str,
              output_file_path: str = input_output.get_data_path(), verbose=False):
    """
    This function combines all functions of the cleaning module to detect and delete outliers and faulty trips
    and writes the cleaned dataframe on disk as a parquet file.
    It can be currently applied to specific months only.
    ----------------------------------------------
    :param
        input_file_path(String): Path to data directory, where the original data set (monthly) is located
        output_file_name(String): The output file name, this parameter should be set by the user
        output_file_path(String): Path to the output directory. Defaults to wd/data.
        month(integer): Month of the year (1 = January, 2 = February...)
        verbose(boolean): Set 'True' to get detailed logging information
    """
    typer.echo('Reading the input file for transformation')
    df = input_output.read_parquet_file(input_file_path)
    typer.echo('Started transforming...')
    df_cleaned = cleaning.clean_dataset(df=df, month=month, verbose=verbose)
    typer.echo('Finished transforming...')
    input_output.write_parquet_file(df=df_cleaned, filename=output_file_name, path=output_file_path)
    typer.echo(f'{output_file_name} successfully saved in {output_file_path}')


@pds_app.command("predict")
def predict(input_file_path: str, output_file_name: str, output_file_path: str = input_output.get_data_path(),
            manhattan: bool = False):
    """
    This function loads the pre-trained models into the memory and predicts fare amount, trip distance, and payment type.
    The resulting prediction are added to the provided data set and written to the disk.
    ----------------------------------------------
    :param
        input_file_path(String): Path to data directory, where the TRANSFORMED data set is located
        output_file_name(String): The prediction file name, this parameter should be set by the user
        output_file_path(String): Path to the output directory. Defaults to wd/data.
        month(integer): Month of the year (1 = January, 2 = February...)
        manhattan(bool): Specifies if the manhattan models should be loaded for prediction.
    """
    typer.echo("Reading the input file for prediction")
    df = input_output.read_parquet_file(input_file_path)
    typer.echo('Initializing the prediction models')
    fare_amount_model, trip_distance_model, payment_type_model = _load_models(manhattan=manhattan)
    fare_amount_relevant_features, trip_distance_relevant_features, payment_type_relevant_features = _get_relevant_features()
    typer.echo('Adding relevant features')
    df_augmented = add_relevant_features(df.copy(), "pickup_datetime")
    typer.echo('Started fare amount prediction')
    _ = _make_data_preparation(df=df_augmented, relevant_features=fare_amount_relevant_features, is_manhattan=manhattan)
    _.pop(fare_amount_relevant_features.get('target'))
    df['Fare_amount_prediction'] = fare_amount_model.predict(_)
    typer.echo('Finished fare amount prediction')
    typer.echo('Started trip distance prediction')
    _ = _make_data_preparation(df=df_augmented, relevant_features=trip_distance_relevant_features, is_manhattan=manhattan)
    _.pop(trip_distance_relevant_features.get('target'))
    df['Trip_distance_prediction'] = trip_distance_model.predict(_)
    typer.echo('Finished trip distance prediction')
    typer.echo('Started payment type prediction')
    _ = _make_data_preparation(df=df_augmented, relevant_features=payment_type_relevant_features,use_created_features=True, is_manhattan=manhattan)
    _.pop(payment_type_relevant_features.get('target'))
    df['Payment_prediction'] = payment_type_model.predict(_)
    typer.echo('Finished payment type prediction')

    input_output.write_parquet_file(df=df, filename=output_file_name, path=output_file_path)


if __name__ == "__main__":
    typer.echo(font.renderText('Welcome to PDS21 CLI -- Created by ANACONDAS'))
    pds_app()
