import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Mumbai House Prices.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    df['price_lakhs'] = df['price'].where(df['price_unit'] == 'L', df['price'] * 100)
    df1 = df.drop(['price', 'price_unit', 'locality'], axis=1)
    
    num_regions = df1['region'].value_counts()
    small_regions = num_regions[num_regions <= 20].index.tolist()
    df1['region'] = df1['region'].apply(lambda region: 'other' if region in small_regions else region)
    
    upper_limit_area = df1['area'].mean() + 3 * df1['area'].std()
    lower_limit_area = df1['area'].mean() - 3 * df1['area'].std()
    df1.loc[df1['area'] > upper_limit_area, 'area'] = upper_limit_area
    df1.loc[df1['area'] < lower_limit_area, 'area'] = lower_limit_area
    df1['area'] = df1['area'].astype(int) 
    upper_limit_price = df1['price_lakhs'].mean() + 3 * df1['price_lakhs'].std()
    lower_limit_price = df1['price_lakhs'].mean() - 3 * df1['price_lakhs'].std()
    df1.loc[df1['price_lakhs'] > upper_limit_price, 'price_lakhs'] = upper_limit_price
    df1.loc[df1['price_lakhs'] < lower_limit_price, 'price_lakhs'] = lower_limit_price
    
    return df1

# Define model pipeline
@st.cache_resource
def train_model(X, y):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bhk', 'area']),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['age', 'status']),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), ['region', 'type'])
        ],
        remainder='passthrough'
    )
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=10)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rf', rf_model)
    ])
    
    pipeline.fit(X, y)
    
    return pipeline

# Load and preprocess data
df = load_data()
df1 = preprocess_data(df)

# Feature selection
X = df1.drop(columns=['price_lakhs'], axis=1)
y = np.log(df1['price_lakhs'])

# Train the model
model = train_model(X, y)

# Streamlit app interface
st.title("Mumbai House Prices Prediction")

# Input form for user data
st.sidebar.header("Input Features")
bhk = st.sidebar.slider("BHK", min_value=1, max_value=6, value=2, step=1)
property_type = st.sidebar.selectbox("Property Type", df1['type'].unique())
area = st.sidebar.number_input("Area (sq ft)", value=1000)
region = st.sidebar.selectbox("Region", df1['region'].unique())
status = st.sidebar.selectbox("Status", df1['status'].unique())
age = st.sidebar.selectbox("Age", df1['age'].unique())

# Prepare the input data
input_data = pd.DataFrame([[bhk, property_type, area, region, status, age]], columns=X.columns)

# Predict the price
predicted_price = np.exp(model.predict(input_data)[0])

st.write(f"### Predicted Price: ₹{predicted_price:.2f} Lakhs")

# Visualization of the dataset
st.write("### Dataset Overview")
st.dataframe(df1.head())

if st.checkbox("Show Histograms"):
    st.write("### Histogram of Features")
    for col in ['bhk', 'area', 'price_lakhs']:
        plt.figure()
        sns.histplot(df1[col], kde=True)
        st.pyplot(plt)

# Display the residuals histogram
if st.checkbox("Show Residual Analysis"):
    y_pred_final = model.predict(X)
    residuals = y - y_pred_final
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Analysis')
    st.pyplot(plt)
