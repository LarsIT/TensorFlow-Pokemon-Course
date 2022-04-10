import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#read the dataset
df = pd.read_csv('pokemon_set.csv')

#removing unnecessary columns from dataframe
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

#It's not possible to calculate with strings or boolean, in order to create graphs I need to convert them into numbers
#Converting a boolean value into numerical boolean, 0 or 1
df['isLegendary'].astype(int)

#in order to stick with the numerical boolean and not create a scale with e.g. 5 = Water; 6 = Ground, I need to create new columns like 'isWater',...,

#create a dummy df with the numerical boolean values for the non-numerical (dummy)columns
#concatenate them into one df
#remove the original columns that were 'dummied'
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df, df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)

df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])


#spllitting data to train a model and test it afterwards
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = DataFrame.drop(column, axis = 1)
    df_test = DataFrame.drop(column, axis = 1)

    return df_train, df_test

df_train, df_test = train_test_splitter(df,'Generation')







