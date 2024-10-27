import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import streamlit as st



levels = pd.read_csv("estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv")
classification = pd.read_csv("Obesity Classification.csv")
dataset = pd.read_csv("ObesityDataSet.csv")

#Print dataset
st.subheader("Dataset Full Table")
st.write(levels)

#Print dataset statistics
st.subheader("Dataset Description")
st.write(levels.describe(include="all"))

#Print dataset information
st.subheader("Dataset Information")
st.write(levels.info())

st.subheader("Conclusion")
st.markdown("The first dataset shows that the data is clean based on the non-existence null entries on each column. This dataset was taken from UCI library. It has been cleaned using techniques: z-score normalization, one-hot encoding, outlier removal, min-max scaling, and feature selection.")
st.markdown("The target column 'NObeyesdad' contains the following encoding of Obesity Levels:")
st.write(""" 
            Insufficient_Weight: 0
         
            Normal_Weight:1

            Overweight_Level_I: 2

            Overweight_Level_II: 3

            Obesity_Type_I: 4

            Obesity_Type_II: 5
         
            Obesity_Type_III: 6""")