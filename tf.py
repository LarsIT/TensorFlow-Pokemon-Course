import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#read the dataset
df = pd.read_csv('pokemon_set.csv')

print(df.columns)