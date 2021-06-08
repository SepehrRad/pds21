import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import yellowcab
from numpy import random


def monthly_visualization_full(df):
    """
    This function visualizes the monthly distribution of trip duration of the given DataFrame with annual trip data,
    including comparison to a normal distribution & returns monthly plots.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame with trip data.
    :returns
        plt.show(): Monthly distribution plots.
    """
    monthsDict = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 12))
    fig.tight_layout(pad=3.0)
    fig.suptitle('Trip duration distribution', fontsize=18)
    plt.subplots_adjust(left=0.1, top=0.9)
    for i, ax in enumerate(ax.flat):
        i = i + 1
        df_m = df.loc[df['pickup_month'] == i]

        l1 = sns.distplot(df_m['trip_duration_minutes'], ax=ax)
        l2 = sns.distplot(random.normal(size=5000, loc=df_m['trip_duration_minutes'].mean()), hist=False, ax=ax)
        l3 = ax.axvline(df_m['trip_duration_minutes'].mean(), linestyle='dashed')

        ax.set_title(monthsDict.get(i), fontsize=12)
        ax.legend([l1, l2, l3],
                  labels=['Original distribution', 'Normal distribution', 'Mean'],
                  loc='upper right',
                  borderaxespad=0.3)
        plt.setp(ax, xlim=(0, 40))
    plt.xlabel('Trip duration (minutes)')
    plt.ylabel('Density')
    plt.show()
