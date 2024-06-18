import streamlit as st
import pandas as pd

@st.cache_data
def load_price_data(fn = "price_data.gzip"):
    price_data = pd.read_parquet(fn)
    return price_data