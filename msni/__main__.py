from argparse import ArgumentParser

import numpy as np

from . import fit_predict


def main():
    parser = ArgumentParser(prog="python -m msni")
    parser.add_argument("features", help="Numerical feature matrix in NumPy .npy format")
    parser.add_argument("labels", help="Output .npy file")
    args = parser.parse_args()
    X = np.load(args.features, allow_pickle=False)
    labels = fit_predict(X)
    np.save(args.labels, labels, allow_pickle=False)


if __name__ == "__main__":
    main()
