import numpy as np
from numpy import ndarray
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os


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
        xy = np.quantile(xy, np.linspace(0, 1, num=num), interpolation="linear")

    # add +ve/-ve inf end points to the parameter value sequence
    xy = np.r_[-np.inf, xy, +np.inf]

    # we build the pp-curve the same way as we build the ROC curve:
    #  by parameterizing with the a monotonic threshold sequence
    #    pp: v \mapsto (\hat{F}_x(v), \hat{F}_y(v))
    #  where \hat{F}_S(v) = \frac1{n_S} \sum_j 1_{S_j \leq v}
    p = np.searchsorted(x, xy) / len(x)
    q = np.searchsorted(y, xy) / len(y)

    return p, q


def filter_presolved(arr):
    return arr[arr > 1]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path")
    parser.add_argument(
        "--key",
        default="num_nodes",
        choices=["num_nodes", "lp_iterations", "solving_time"],
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)
    key = args.key
    path = args.path
    strong = filter_presolved(pd.read_csv(f"{path}/strong.csv")[key].to_numpy())

    for fname in os.listdir(path):
        if not fname.endswith(".csv"):
            continue

        data = filter_presolved(pd.read_csv(f"{path}/" + fname)[key].to_numpy())
        print(
            f"{fname}: tot = {len(data)}, median = {np.median(data):.2f}, "
            f"mean = {np.mean(data):.2f}, std = {np.std(data):.2f}"
        )

        # plt.hist(data, bins=100, log=True)
        # plt.title(fname)
        # plt.savefig(f'hist_{fname[:-4]}.pdf')
        # plt.close()

        u, p = pp_curve(x=strong, y=data)
        ax.plot(u, p, label=fname[:-4])

    ax.plot((0, 1), (0, 1), c="k", zorder=10, alpha=0.25)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect(1.0)
    ax.set_title(key)

    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect(1.0)

    plt.legend()
    plt.savefig(f"pp_{key}.pdf")
    plt.show()
