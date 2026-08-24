#!/usr/bin/env python3

from os import sep
import scipy.stats
import numpy as np

def load():
    import pandas as pd
    return pd.read_csv("src/iris.csv").drop('species', axis=1).values

def lengths():
    pass

def correlations():
    pass

def main():
    print(lengths())
    print(correlations())

if __name__ == "__main__":
    main()
