#######################
# Import libraries
import streamlit as st
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

st.header("Welcome to Janice Data Science Project Webpage!")

st.subheader("Introduction")

st.subheader("Meet our team members!")
st.markdown("Janice Ivana")
st.image("jan.jpg")
st.write("Hi there! I'm a fresh data science graduate from Monash University, this webpage will be my main portofolio to show my work I have done so far :)")

levels = pd.read_csv("estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv")
classification = pd.read_csv("Obesity Classification.csv")
dataset = pd.read_csv("ObesityDataSet.csv")

st.write(levels.head())

