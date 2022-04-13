import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the dataset
df = pd.read_csv('pokemon_set.csv')

#removing unnecessary columns from dataframe
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]



#changing string values into numerical boolean, with dummies
def dummy_creation(df):
    columns = df.columns
    dummy_categories = []
    
    #checks for object dtypes and appends their columns to dummy_categories
    for i in columns:
        if df[i].dtype == object:
            dummy_categories.append(i)
    
    i = 0
    #creates another table with the object columns but now as 'uint8'-dtypes, concatenates them
    #and removes the original object columns
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df, df_dummy],axis=1)
        df = df.drop(i, axis=1)

    return(df)    

df = dummy_creation(df)
print(df.dtypes)

#what if i dont have to put in the strings manually,
#i have to make the function check the dtype of every columns first entry
#finding strings/objects

#df.dtypes

#df.columns     creates a list of the columns, each column can be accessed through normal list index numbers

#print(df.iloc[0][1]) #puts out value of [row],[column]

