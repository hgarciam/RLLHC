import sys
import os.path
import RLLHC
import pandas as pd
from time import time
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

modelfile = 'Ridge_surrogate_20k.pkl'
estimator = pickle.load(open(modelfile, "rb"))

test_inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prediction = estimator.predict(np.reshape(test_inputs, (1, -1)))

print(prediction)