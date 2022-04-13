import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the dataset
df = pd.read_csv('pokemon_set.csv')

#removing unnecessary columns from dataframe
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

df.head().to_csv()