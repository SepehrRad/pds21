# PDS Project 2021
Programming Data Science Project "Yellow Cab".
University of Cologne - Summer Semester 2021
## Group

* Nina Annika Erlacher
* Simon Maximilian Wolf
* Christian Bergen
* Nico Liedmeyer 
* Sepehr Salemirad

## Setup

We recommend to setup a virtual environment or conda environment with python version 3.8 and installing jupyter notebook via pip:

```
pip install notebook
```

Download the cleaned data set from:

Put the cleaned data into data/input/cleaned to run our Notebook Demo.

Running the notebook for the first time will install all necessary packages. You don't have to do anything else.

If you do not want to use the Demo, we recommend running the following command in a seperate notebook:

```
!pip install -e ..
```
    
## Package descriptions
* **cleaning:** Removing invalid data and outliers from raw trip dataset.
* **eda:** Exploration, aggregation and plotting functions, including dashboard functionality.
* **feature_engineering:** Adding useful columns to the trip dataset for further analysis.
* **input_output:** Reading and writing .parquet, .geojson and .pickle files. Also provides utility functionality.
* **model:** Building, benchmarking and optimizing predictive models.
* **prediction:** Preparing data, selecting features, setting up a pipeline and validating the results.
* **preprocessing:** Transforming different types of dataframe columns.

## CLI

yellowcab also provides a CLI to clean/transform data aswell prediciting fare amount, trip distance and payment type with pre-trained models. To use the CLI go to the yellowcab directory in your console and run one of the following commands.

To clean and transform data:

``

## Troubleshooting

### Cleaning

#### detecting all data as outlier

Please update pandas( >= v.1.2.4) and numpy(>=1.20.2)

### Dashboard

#### Version

#### Maps dont show up

Just click on the tiles symbole in the upper right corner.

#### Dashboard doesn't start, while running Notebook_demo

Wait for the notebook to finish running all cells. Afterwards go to the corresponding localhost.
