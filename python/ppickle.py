#!/usr/bin/env python3
from __future__ import print_function

import sys
import pandas as pd
import pickle

path = sys.argv[1]
obj  = pickle.load(open(path))

pd.set_option('display.width', 800)
print(obj)
