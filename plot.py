import numpy as np
from numpy import ndarray
import pandas as pd
from matplotlib import pyplot as plt


def pp_curve(*, x: ndarray, y: ndarray, num: int = None) -> tuple[ndarray, ndarray]:
    """Build threshold-parameterized pipi curve."""
    # sort each sample for fast O(\log n) eCDF queries by `searchsorted`
    x, y = np.sort(x), np.sort(y)

    # pool sorted samples to get thresholds
    xy = np.concatenate((x, y))
    if num is None:
        # finest detail thresholds: sort the pooled samples (sorted
        #  arrays can be merged in O(n), but it turns out numpy does
        #  not have the procedure)
        xy.sort()

    else:
        # coarsen by finding threshold grid in the pooled sample, that
        #  is equispaced after being transformed by the empirical cdf.
        xy = np.quantile(xy, np.linspace(0, 1, num=num), interpolation='linear')

    # add +ve/-ve inf end points to the parameter value sequence
    xy = np.r_[-np.inf, xy, +np.inf]

    # we build the pp-curve the same way as we build the ROC curve:
    #  by parameterizing with the a monotonic threshold sequence
    #    pp: v \mapsto (\hat{F}_x(v), \hat{F}_y(v))
    #  where \hat{F}_S(v) = \frac1{n_S} \sum_j 1_{S_j \leq v}
    p = np.searchsorted(x, xy) / len(x)
    q = np.searchsorted(y, xy) / len(y)

    return p, q


def filter(arr):
    return arr[arr > 1]


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    key = 'num_nodes'
    strong = filter(pd.read_csv('results/strong_branching.csv')[key].to_numpy())

    for fname in [
        'random.csv',
        'dqn.csv',
        'dqn_v2.csv',
        'dqn_v2_start.csv',
        'retro/retro.csv',
        'retro/il.csv',
        'retro/strong_branching.csv',
        'retro/random.csv'
    ]:
        data = filter(pd.read_csv('results/' + fname)[key].to_numpy())
        print(f'{fname}: median = {np.median(data)}, mean = {np.mean(data)}, std = {np.std(data)}')

        # plt.hist(data, bins=100, log=True)
        # plt.title(fname)
        # plt.savefig(f'hist_{fname[:-4]}.pdf')
        # plt.close()

        u, p = pp_curve(x=strong, y=data)
        ax.plot(u, p, label=fname)

    ax.plot((0, 1), (0, 1), c="k", zorder=10, alpha=0.25)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect(1.)
    ax.set_title(key)

    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect(1.)

    plt.legend()
    plt.savefig('pp_nodes.pdf')
    plt.show()
