import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import yellowcab.eda.aggregation


def basic_plots(df):

    df_agg_m = yellowcab.eda.agg_stats(df['pickup_month'], df['pickup_month'], ["count"])

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 12))
    fig.tight_layout(pad=3.0)
    fig.suptitle('Basic plots', fontsize=18)

    axs[0].plot(x=df_agg_m[0], y=df_agg_m[1])

