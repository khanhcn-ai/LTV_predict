import pandas as pd
import numpy as np
import os
from modeling.train_model_per_day import *
import warnings
warnings.filterwarnings('ignore', message="X does not have valid feature names")

list_fail = [7, 51, 54, 55]

for i in range(24, 57, 2):
    build_model_per_day(i)