import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
import pycountry
import plotly.express as px
from termcolor import colored
import squarify
import folium
from folium import plugins
import plotly.graph_objects as go
from scipy import stats

st.title("Hello World")


df = pd.read_csv("earthquake_1995-2023.csv")
df.head(5)

