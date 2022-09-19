import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

model = joblib.load("model.pkl")

overal_qual = st.slider(
    "Overall material and finish quality", min_value=1, max_value=10, value=6
)
st.write(overal_qual)
total_bsmt_sf = st.slider(
    "Total square feet of basement area", min_value=0, max_value=6110, value=1045
)
st.write(total_bsmt_sf)
floor_1st_sf = st.slider(
    "First Floor square feet", min_value=400, max_value=4692, value=1500
)
st.write(floor_1st_sf)
gr_liv_area = st.slider(
    "Above grade (ground) living area square feet",
    min_value=400,
    max_value=5642,
    value=1500,
)
st.write(gr_liv_area)
garage_cars = st.slider(
    "Size of garage in car capacity", min_value=0, max_value=5, value=2
)
st.write(garage_cars)
garage_area = st.slider(
    "Size of garage in square feet", min_value=0, max_value=1418, value=500
)
st.write(garage_area)

estimation_button = st.button("Click to estimate price")

if estimation_button:
    st.write(
        f"Price: {model.predict(np.array([overal_qual, total_bsmt_sf, floor_1st_sf, gr_liv_area, garage_cars, garage_area]).reshape(1, -1))[0]}"
    )
