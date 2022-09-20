import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.DataFrame()
if "mdf" not in st.session_state:
    st.session_state.mdf = df
model = joblib.load("model.pkl")


if "overal_qual" not in st.session_state:
    st.session_state["overal_qual"] = 6

overal_qual = st.slider(
    "Overall material and finish quality", min_value=1, max_value=10, value=6
)
st.session_state.overal_qual = overal_qual

if "total_bsmt_sf" not in st.session_state:
    st.session_state["total_bsmt_sf"] = 100

total_bsmt_sf = st.slider(
    "Total square meters of basement area", min_value=0, max_value=550, value=100
)
st.session_state.total_bsmt_sf = total_bsmt_sf

if "floor_1st_sf" not in st.session_state:
    st.session_state["floor_1st_sf"] = 100

floor_1st_sf = st.slider(
    "First Floor square meters", min_value=30, max_value=400, value=100
)
st.session_state.floor_1st_sf = floor_1st_sf

if "gr_liv_area" not in st.session_state:
    st.session_state["gr_liv_area"] = 140


gr_liv_area = st.slider(
    "Above grade (ground) living area square meters",
    min_value=30,
    max_value=500,
    value=140,
)
st.session_state.gr_liv_area = gr_liv_area

if "garage_cars" not in st.session_state:
    st.session_state["garage_cars"] = 2


garage_cars = st.slider(
    "Size of garage in car capacity", min_value=0, max_value=5, value=2
)
st.session_state.garage_cars = garage_cars

if "garage_area" not in st.session_state:
    st.session_state["garage_area"] = 40


garage_area = st.slider(
    "Size of garage in square meters", min_value=0, max_value=130, value=40
)
st.session_state.garage_area = garage_area


estimation_button = st.button("Click to estimate price")

if estimation_button:
    predicted_price = int(
        model.predict(
            np.array(
                [
                    overal_qual,
                    total_bsmt_sf,
                    floor_1st_sf,
                    gr_liv_area,
                    garage_cars,
                    garage_area,
                ]
            ).reshape(1, -1)
        )[0]
    )

    df_new = pd.DataFrame(
        {
            "Overal quality": [overal_qual],
            "Total basement area, m2": [total_bsmt_sf],
            "First floor, m2": [floor_1st_sf],
            "Living area, m2": [gr_liv_area],
            "Garage cars": [garage_cars],
            "Garage area, m2": [garage_area],
            "Sale price, $": [predicted_price],
        }
    )
    st.session_state.mdf = pd.concat([st.session_state.mdf, df_new], axis=0)
    st.write(f"Price: {predicted_price}$")
if len(st.session_state.mdf) > 0:
    st.session_state.mdf.reset_index(drop=True, inplace=True)
    st.dataframe(st.session_state.mdf)
    st.download_button(
        label="Download data as CSV",
        data=st.session_state.mdf.to_csv().encode("utf-8"),
        file_name="houses.csv",
        mime="text/csv",
    )
