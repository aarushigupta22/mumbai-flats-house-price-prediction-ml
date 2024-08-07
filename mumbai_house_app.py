import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("Mumbai House Prices.csv")

X = df[['bhk', 'area', 'type', 'region', 'status', 'age']]  # Example feature set
y = df['price_lakhs']  # Target variable