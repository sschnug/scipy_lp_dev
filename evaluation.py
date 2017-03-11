import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def data_wrangling(results):
    # solvers = [set(x[1]) for x in results]
    # instances = [set(x[0] for x in results)]

    df = pd.DataFrame(results,
                      columns=['instance', 'solver', 'valid', 'time'])

    return df

def plot_overview(df):
    f, axarr = plt.subplots(4)

    """ robustness -> heatmap """
    success = df.pivot('instance', 'solver', 'valid')
    sns.heatmap(success, cbar=False, cmap='RdYlGn', linewidths=.5, ax=axarr[0])
    axarr[0].set_title('Robustness-eval: objective valid withon tol?')
    axarr[0].set_ylabel('Instance')
    axarr[0].set_xlabel('Solver')
    axarr[0].yaxis.set_ticks_position('none')

    """ robustness -> count """
    sns.barplot(x='solver', y='valid', data=df, estimator=sum, ci=None, ax=axarr[1])
    axarr[1].set_title('Robustness-eval: number of solves instances')
    axarr[1].set_ylabel('# Solved')
    axarr[1].set_xlabel('Solver')
    axarr[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    """ absolute time -> barplot """
    # set time to zero if invalid => only for visualization
    df['time'] = df.apply(lambda row: 0 if not row['valid'] else row['time'], axis=1)
    sns.barplot(x="instance", y="time", hue="solver", data=df, ax=axarr[2])
    axarr[2].set_title('Performance-eval: absolute time used')
    axarr[2].set_ylabel('Time (s)')
    axarr[2].set_xlabel('Instance')

    """ relative time -> barplot """
    df['time'] /= df.groupby('instance')["time"].transform(max)
    sns.barplot(x="instance", y="time", hue="solver", data=df, ax=axarr[3])
    axarr[3].set_title('Performance-eval: relative time (to slowest) used')
    axarr[3].set_ylabel('Relative Time')
    axarr[3].set_xlabel('Instance')

    plt.tight_layout()
    plt.show()
