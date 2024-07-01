# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import numpy as np
from minisom import MiniSom
import pickle

#------------------------------------------------------------------------------------#
# Details of project
st.title('Earthquake Clustering')
st.title('  ')

st.header('Feature Selection:')
st.write('1. Magnitude: The magnitude of the earthquake.')
st.write('2. Community Internet Intensity Map (CDI): The maximum reported intensity for the event range.')
st.write('3. Modified Mercalli Intensity (MMI): The maximum estimated instrumental intensity for the event.')
st.write('4. Tsunami: "1" for events in oceanic regions and "0" otherwise.')
st.write('5. Significant Event ID (SIG): A number describing how significant the event is. Larger numbers indicate a more significant event.')
st.write('6. Minimum Distance to Station (DMIN): Horizontal distance from the epicenter to the nearest station.')
st.write('7. Azimuthal Gap (GAP): The largest azimuthal gap between azimuthally adjacent stations (in degrees).')
st.write('8. Depth: The depth where the earthquake begins to rupture. ')
st.title('  ')
st.title('  ')

#------------------------------------------------------------------------------------#
# Earthquake Location and Impact Visualization

# Load the earthquake data
df = pd.read_csv("earthquake_data_no_outliers.csv")

st.title("Data Visualization")
st.title("   ")

# Plot a figure to show the location and impact of earthquakes
dff1 = df.copy()
dff1.loc[:, 'magnitude_level'] = pd.cut(dff1['magnitude'], bins=[6.5, 7, 7.5, 8], labels=['6.5 - 7.0', '7.0 - 7.5', '7.5 - 8.0'])

fig1 = px.scatter_geo(dff1, lat='latitude', lon='longitude',
                      hover_name='title', size='sig',
                      color='magnitude_level',
                      color_discrete_map={
                          '6.5 - 7.0': 'green',
                          '7.0 - 7.5': 'orange',
                          '7.5 - 8.0': 'red',
                      },
                      labels={'magnitude_level': 'Magnitude Level'})

fig1.update_geos(projection_type="natural earth", showcountries=True, showcoastlines=True)
# fig1.update_layout(title="Visualizing Earthquake Location and Impact")

# Display the Plotly figure in Streamlit
st.header("Visualizing Earthquake Location and Impact")
st.plotly_chart(fig1)

#------------------------------------------------------------------------------------#

# Tsunami Location and Impact Visualization

# Plot a figure to show the location and impact of tsunamis
dff1 = df.copy()

# Map values in the 'tsunami' column to 'Yes' and 'No'
dff1['tsunami'] = dff1['tsunami'].map({1: 'Yes', 0: 'No'})

# Plot a figure to show the location and impact of tsunamis
fig2 = px.scatter_geo(dff1, lat='latitude', lon='longitude',
                      hover_name='title', size='sig',
                      color='tsunami',
                      color_discrete_map={
                          'Yes': 'navy',      # Color for tsunami = Yes (Tsunami occurred)
                          'No': 'peachpuff'        # Color for tsunami = No (No tsunami)
                      },
                      labels={'tsunami': 'Tsunami (No/Yes)'})

fig2.update_geos(projection_type="natural earth", showcountries=True, showcoastlines=True)
# fig2.update_layout(title="Visualizing Tsunami Location and Occurrence")

# Display the Plotly figure in Streamlit
st.title("   ")
st.header("Visualizing Tsunami Location and Occurrence")
st.plotly_chart(fig2)

#------------------------------------------------------------------------------------#

# Number of Tremors By Year

# Adding year column and creating Other location ('minor activity countries')
ddd = df.copy()
ddd['location'] = ddd['location'].str.split(', ').str[-1]
s = ddd['location'].value_counts().sort_values(ascending=False).reset_index()
top_15 = s.iloc[:15, 0].to_list()
ddd['year'] = pd.to_datetime(ddd['date_time']).dt.year
ddd['location'] = ddd['location'].apply(lambda x: 'Other' if x not in top_15 else x)

time = ddd.groupby(['location', 'year']).agg('count').reset_index()[['location', 'year', 'title']]
to_print = ['World', 'Indonesia', 'Minor Activity Countries', 'Japan', 'Papua New Guinea'] # columns to be printed

# Contains all years from 1995 - 2023
all_years = [x for x in range(1995, 2024)]

# Creating Indonesia and Other columns
indonesia = time[time['location'] == 'Indonesia'].set_index('year')
other = time[time['location'] == 'Other'].set_index('year')
japan = time[time['location'] == 'Japan'].set_index('year').reindex(all_years, fill_value=0)
papua = time[time['location'] == 'Papua New Guinea'].set_index('year').reindex(all_years, fill_value=0)

world = ddd.groupby('year').agg('count')[['title']]
world['Indonesia'] = indonesia['title']
world['Other'] = other['title']
world['Japan'] = japan['title']
world['Papua New Guinea'] = papua['title']

world.columns = to_print
world = world.reset_index()

# Streamlit app
st.title("   ")
st.header("Number of Tremors by Year")

# Plot the Number of tremors by year
fig1 = px.line(world, x='year', y=to_print,
               title='Number of tremors by year (World, Indonesia, Japan, Papue New Guinea, and Minor Activity Countries)')

fig1.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(showgrid=False, title='Year'),
    yaxis=dict(showgrid=False, title='Number of Tremors')
)

line_colors = ['#236952', '#800020', '#ABA0D9', '#002FA7', '#69A196']

for i, color in enumerate(line_colors):
    fig1.data[i].line.color = color

st.plotly_chart(fig1)


#------------------------------------------------------------------------------------#

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA

# # Load data
# df_outliers = pd.read_csv("earthquake_data_outliers.csv")
# df_no_outliers = pd.read_csv("earthquake_data_no_outliers.csv")

# # Select features for clustering
# clustering_data_1 = df_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]
# clustering_data_2 = df_no_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]

# # Scale the data
# scaler = MinMaxScaler()
# minmax_data_scaled_1 = scaler.fit_transform(clustering_data_1)
# minmax_data_scaled_2 = scaler.fit_transform(clustering_data_2)

#------------------------------------------------------------------------------------#

# # K-Means
# st.title('  ')
# st.title('  ')
# st.title("K-Means")

# # Calculate WCSS for KMeans
# def calculate_wcss(data):
#     wcss = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#         kmeans.fit(data)
#         wcss.append(kmeans.inertia_)
#     return wcss

# # # Plot Elbow Method for both datasets
# # st.title("   ")
# # st.header('Elbow Method (K-Means)')

# # # Dataframe with Outliers
# # st.subheader("Dataframe with Outliers")
# # wcss_1 = calculate_wcss(minmax_data_scaled_1)
# # fig1, ax1 = plt.subplots()
# # ax1.plot(range(1, 11), wcss_1, marker='o', linestyle='--')
# # ax1.set_title('Elbow Method (Dataframe with Outliers)')
# # ax1.set_xlabel('Number of Clusters')
# # ax1.set_ylabel('WCSS')
# # ax1.grid(True)
# # st.pyplot(fig1)

# # # Dataframe without Outliers
# # st.subheader("Dataframe without Outliers")
# # wcss_2 = calculate_wcss(minmax_data_scaled_2)
# # fig2, ax2 = plt.subplots()
# # ax2.plot(range(1, 11), wcss_2, marker='o', linestyle='--')
# # ax2.set_title('Elbow Method (Dataframe without Outliers)')
# # ax2.set_xlabel('Number of Clusters')
# # ax2.set_ylabel('WCSS')
# # ax2.grid(True)
# # st.pyplot(fig2)

# # # Silhouette Method
# # def calculate_silhouette(data):
# #     silhouette_scores = []
# #     for i in range(2, 11):
# #         kmeans = KMeans(n_clusters=i, random_state=42)
# #         kmeans.fit(data)
# #         labels = kmeans.labels_
# #         silhouette_avg = silhouette_score(data, labels)
# #         silhouette_scores.append(silhouette_avg)
# #     return silhouette_scores

# # # Plot Silhouette Method for both datasets
# # st.title("   ")
# # st.header('Silhouette Method (K-Means)')

# # # Dataframe with Outliers
# # st.subheader("Dataframe with Outliers")
# # silhouette_scores_1 = calculate_silhouette(minmax_data_scaled_1)
# # fig3, ax3 = plt.subplots()
# # ax3.plot(range(2, 11), silhouette_scores_1, marker='o', linestyle='--')
# # ax3.set_title('Silhouette Method (Dataframe with Outliers)')
# # ax3.set_xlabel('Number of Clusters')
# # ax3.set_ylabel('Silhouette Score')
# # ax3.grid(True)
# # st.pyplot(fig3)

# # # Dataframe without Outliers
# # st.subheader("Dataframe without Outliers")
# # silhouette_scores_2 = calculate_silhouette(minmax_data_scaled_2)
# # fig4, ax4 = plt.subplots()
# # ax4.plot(range(2, 11), silhouette_scores_2, marker='o', linestyle='--')
# # ax4.set_title('Silhouette Method (Dataframe without Outliers)')
# # ax4.set_xlabel('Number of Clusters')
# # ax4.set_ylabel('Silhouette Score')
# # ax4.grid(True)
# # st.pyplot(fig4)

# # KMeans Clustering and Visualization
# def kmeans_clustering(data, title):
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     labels = kmeans.fit_predict(data)

#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(data)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for cluster_num in range(2):
#         subset = pca_result[labels == cluster_num]
#         ax.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

#     ax.set_title(title)
#     ax.set_xlabel('PCA 1')
#     ax.set_ylabel('PCA 2')
#     ax.legend()
#     ax.grid(True)
#     st.pyplot(fig)

# # KMeans Clustering with Visualization for both datasets
# st.title("   ")
# st.header('K-means Clustering (PCA visualization)')

# # Dataframe with Outliers
# st.subheader("Dataframe with Outliers")
# kmeans_clustering(minmax_data_scaled_1, "K-means Clustering with 2 Clusters (PCA visualization) - Dataframe with Outliers")

# # Dataframe without Outliers
# st.subheader("Dataframe without Outliers")
# kmeans_clustering(minmax_data_scaled_2, "K-means Clustering with 2 Clusters (PCA visualization) - Dataframe without Outliers")

# # Calculate silhouette scores for KMeans clustering
# kmeans1 = KMeans(n_clusters=2, random_state=42)
# labels_1 = kmeans1.fit_predict(minmax_data_scaled_1)
# kmeans1_silhouette = silhouette_score(minmax_data_scaled_1, labels_1, metric='euclidean')

# kmeans2 = KMeans(n_clusters=2, random_state=42)
# labels_2 = kmeans2.fit_predict(minmax_data_scaled_2)
# kmeans2_silhouette = silhouette_score(minmax_data_scaled_2, labels_2, metric='euclidean')

# # Display silhouette scores
# st.title("   ")
# st.header('Silhouette Scores')
# st.write("KMeans1 (Dataframe with Outliers):", kmeans1_silhouette)
# st.write("KMeans2 (Dataframe without Outliers):", kmeans2_silhouette)

# #------------------------------------------------------------------------------------#

# # GMM

# st.title("   ")
# st.title("   ")
# st.title("GMM")

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA

# # Load data
# df_outliers = pd.read_csv("earthquake_data_outliers.csv")
# df_no_outliers = pd.read_csv("earthquake_data_no_outliers.csv")

# # Select features for clustering
# clustering_data_1 = df_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]
# clustering_data_2 = df_no_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]

# # Scale the data
# scaler = MinMaxScaler()
# minmax_data_scaled_1 = scaler.fit_transform(clustering_data_1)
# minmax_data_scaled_2 = scaler.fit_transform(clustering_data_2)

# # # Calculate BIC scores for Gaussian Mixture Model
# # def calculate_bic(data):
# #     bic_scores = []
# #     for n_components in range(2, 11):
# #         gmm = GaussianMixture(n_components=n_components, random_state=42)
# #         gmm.fit(data)
# #         bic_scores.append(gmm.bic(data))
# #     return bic_scores

# # # Plot BIC scores for Gaussian Mixture Model
# # st.title("   ")
# # st.header('BIC Scores for Gaussian Mixture Model')

# # # Dataframe with Outliers
# # st.subheader("Dataframe with Outliers")
# # bic_scores_1 = calculate_bic(minmax_data_scaled_1)
# # fig1, ax1 = plt.subplots()
# # ax1.plot(range(2, 11), bic_scores_1, marker='o')
# # ax1.set_xlabel('Number of components')
# # ax1.set_ylabel('BIC Score')
# # ax1.set_title('BIC Scores (Dataframe with Outliers)')
# # ax1.grid(True)
# # st.pyplot(fig1)

# # # Dataframe without Outliers
# # st.subheader("Dataframe without Outliers")
# # bic_scores_2 = calculate_bic(minmax_data_scaled_2)
# # fig2, ax2 = plt.subplots()
# # ax2.plot(range(2, 11), bic_scores_2, marker='o')
# # ax2.set_xlabel('Number of components')
# # ax2.set_ylabel('BIC Score')
# # ax2.set_title('BIC Scores (Dataframe without Outliers)')
# # ax2.grid(True)
# # st.pyplot(fig2)

# # Perform Gaussian Mixture Model clustering
# def gmm_clustering(data, title):
#     gmm = GaussianMixture(n_components=2, random_state=42)
#     labels = gmm.fit_predict(data)

#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(data)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for cluster_num in range(2):
#         subset = pca_result[labels == cluster_num]
#         ax.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

#     ax.set_title(title)
#     ax.set_xlabel('PCA 1')
#     ax.set_ylabel('PCA 2')
#     ax.legend()
#     ax.grid(True)
#     st.pyplot(fig)

# # Gaussian Mixture Model Clustering with Visualization for both datasets
# st.title("   ")
# st.header('GMM Clustering')

# # Dataframe with Outliers
# st.subheader("Dataframe with Outliers")
# gmm_clustering(minmax_data_scaled_1, "Gaussian Mixture Model Clustering with 2 Components (PCA visualization) - Dataframe with Outliers")

# # Dataframe without Outliers
# st.subheader("Dataframe without Outliers")
# gmm_clustering(minmax_data_scaled_2, "Gaussian Mixture Model Clustering with 2 Components (PCA visualization) - Dataframe without Outliers")

# # Calculate silhouette scores for Gaussian Mixture Model clustering
# gmm1 = GaussianMixture(n_components=2, random_state=42)
# labels_1 = gmm1.fit_predict(minmax_data_scaled_1)
# gmm1_silhouette = silhouette_score(minmax_data_scaled_1, labels_1, metric='euclidean')

# gmm2 = GaussianMixture(n_components=2, random_state=42)
# labels_2 = gmm2.fit_predict(minmax_data_scaled_2)
# gmm2_silhouette = silhouette_score(minmax_data_scaled_2, labels_2, metric='euclidean')

# # Display silhouette scores
# st.title("   ")
# st.header('Silhouette Scores')
# st.write("Gaussian Mixture Model 1 (Dataframe with Outliers):", gmm1_silhouette)
# st.write("Gaussian Mixture Model 2 (Dataframe without Outliers):", gmm2_silhouette)

# #------------------------------------------------------------------------------------#

# # DBSCAN

# st.title("   ")
# st.title("   ")
# st.title("DBSCAN")

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# from sklearn.decomposition import PCA

# # Load data
# df_outliers = pd.read_csv("earthquake_data_outliers.csv")
# df_no_outliers = pd.read_csv("earthquake_data_no_outliers.csv")

# # Select features for clustering
# clustering_data_1 = df_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]
# clustering_data_2 = df_no_outliers[["magnitude", "cdi", "mmi", "tsunami", "sig", "dmin", "gap", "depth"]]

# # Scale the data
# scaler = MinMaxScaler()
# minmax_data_scaled_1 = scaler.fit_transform(clustering_data_1)
# minmax_data_scaled_2 = scaler.fit_transform(clustering_data_2)

# # Function to perform grid search for DBSCAN parameters
# def grid_search_dbscan(data_scaled):
#     eps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
#     min_samples_values = [5, 10, 15, 20]

#     best_score = -1
#     best_eps = None
#     best_min_samples = None

#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             cluster_labels = dbscan.fit_predict(data_scaled)
#             silhouette_avg = silhouette_score(data_scaled, cluster_labels)
#             print(f"For eps={eps}, min_samples={min_samples}, silhouette score: {silhouette_avg}")

#             if silhouette_avg >= best_score:
#                 best_score = silhouette_avg
#                 best_eps = eps
#                 best_min_samples = min_samples

#     print(f"\nBest parameters: eps={best_eps}, min_samples={best_min_samples}, silhouette score: {best_score}")

#     return best_eps, best_min_samples, best_score

# # Function to perform DBSCAN clustering and visualization
# def visualize_dbscan_clusters(data_scaled, df_clusters, num_clusters, title):
#     eps, min_samples, silhouette_score = grid_search_dbscan(data_scaled)

#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     df_clusters['DBSCAN_Cluster'] = dbscan.fit_predict(data_scaled)

#     pca = PCA(n_components=2)
#     data_pca = pca.fit_transform(data_scaled)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for cluster_num in set(df_clusters['DBSCAN_Cluster']):
#         if cluster_num == -1:
#             subset = data_pca[df_clusters['DBSCAN_Cluster'] == cluster_num]
#             ax.scatter(subset[:, 0], subset[:, 1], label=f"Outliers", alpha=0.6, color='black')
#         else:
#             subset = data_pca[df_clusters['DBSCAN_Cluster'] == cluster_num]
#             ax.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

#     ax.set_title(title)
#     ax.set_xlabel('PCA 1')
#     ax.set_ylabel('PCA 2')
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     st.pyplot(fig)

#     return silhouette_score

# # Streamlit app
# st.title("   ")
# st.header("DBSCAN Clustering Visualization")

# # With outliers
# st.subheader('With Outliers')
# dbscan1_silhouette = visualize_dbscan_clusters(minmax_data_scaled_1, df_outliers.copy(), num_clusters=2, title='DBSCAN Clustering (PCA Visualization & Dataframe with Outliers)')

# # Without outliers
# st.subheader('Without Outliers')
# dbscan2_silhouette = visualize_dbscan_clusters(minmax_data_scaled_2, df_no_outliers.copy(), num_clusters=2, title='DBSCAN Clustering (PCA Visualization & Dataframe without Outliers)')

# # Print silhouette scores
# st.title("   ")
# st.header('Silhouette Scores')
# st.write("DBSCAN Silhouette Score (With Outliers):", dbscan1_silhouette)
# st.write("DBSCAN Silhouette Score (Without Outliers):", dbscan2_silhouette)

# #------------------------------------------------------------------------------------#

# # Model Performance Comparision

# st.title("  ")
# st.title("  ")
# st.title('Model Performance Compare')

# import streamlit as st
# import matplotlib.pyplot as plt

# # Assuming you have these scores stored in variables
# scores = {
#     'KMeans1': kmeans1_silhouette,
#     'KMeans2': kmeans2_silhouette,
#     'GMM1': gmm1_silhouette,
#     'GMM2': gmm2_silhouette,
#     'DBSCAN1': dbscan1_silhouette,
#     'DBSCAN2': dbscan2_silhouette,
#     'KMedoidsMin1': kmedoidsmin1_silhouette,
#     'KMedoidsMin2': kmedoidsmin2_silhouette
# }

# # Extracting names and scores
# names = list(scores.keys())
# values = list(scores.values())

# # Define custom colors for each algorithm
# custom_colors = {
#     'KMeans1': 'indianred',
#     'KMeans2': 'pink',
#     'GMM1': 'dodgerblue',
#     'GMM2': 'skyblue',
#     'DBSCAN1': 'navajowhite',
#     'DBSCAN2': 'antiquewhite',
#     'KMedoidsMin1': 'mediumseagreen',
#     'KMedoidsMin2': 'lightgreen'
# }

# # Plotting with custom colors
# fig, ax = plt.subplots(figsize=(9, 6))
# bars = ax.bar(names, values, color=[custom_colors[name] for name in names])
# ax.set_xlabel('Clustering Algorithms')
# ax.set_ylabel('Silhouette Score')
# ax.set_title('Silhouette Scores of Different Clustering Algorithms')
# ax.set_xticklabels(names, rotation=45, ha='right')

# # Displaying the plot in Streamlit
# st.pyplot(fig)

#------------------------------------------------------------------------------------#

# Predict Cluster

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
df_outliers = pd.read_csv("earthquake_data_outliers.csv")
df_no_outliers = pd.read_csv("earthquake_data_no_outliers.csv")

clustering_data_1 = df_outliers[["magnitude", "cdi", "mmi", "tsunami",
                                 "sig", "dmin", "gap", "depth"]]
clustering_data_2 = df_no_outliers[["magnitude", "cdi", "mmi", "tsunami",
                                    "sig", "dmin", "gap", "depth"]]

cluster_vars = ["magnitude", "cdi", "mmi", "tsunami",
                "sig", "dmin", "gap", "depth"]

# Scale data
minmax_scaler = MinMaxScaler()
minmax_data_scaled_1 = minmax_scaler.fit_transform(clustering_data_1)
minmax_data_scaled_2 = minmax_scaler.transform(clustering_data_2)

# KMeans Clustering
kmeans1 = KMeans(n_clusters=2, random_state=42)
clustering_data_1['Cluster_KMEANS_O'] = kmeans1.fit_predict(minmax_data_scaled_1)

# PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(minmax_data_scaled_1)

# Mapping dictionary
cluster_mapping = {0: 'High Tsunami Risk Seismic Activity',
                   1: 'Moderate Magnitude Seismic Activity'}

# Streamlit App
st.sidebar.title('Earthquake Cluster Predict')
st.title('  ')

# st.sidebar.header('Input Data')

# User Input for prediction
magnitude = st.sidebar.text_input('Earthquake magnitude (1.0 to 10.0)', value='')
cdi = st.sidebar.text_input('Community Decimal Intensities(0 to 9)', value='')
mmi = st.sidebar.text_input('Modified Mercalli Intensity (0 to 9)', value='')
tsunami = st.sidebar.selectbox('Tsunami(0 = False, 1 = True)', [0, 1])
sig = st.sidebar.text_input('Significant Event ID (100 to 3000)', value='')
dmin = st.sidebar.text_input('Minimum Distance to Station (1.0 to 20.0) ', value='')
gap = st.sidebar.text_input('Azimuthal gap (1 to 250)', value='')
depth = st.sidebar.text_input('Depth of Rupture (1 to 1000)', value='')

def validate_input(value, min_value, max_value):
    try:
        value = float(value)
        if value < min_value or value > max_value:
            return False, f"Value should be between {min_value} and {max_value}."
        return True, value
    except ValueError:
        return False, "Invalid input. Please enter a numeric value."

if st.sidebar.button('Predict'):
    valid = True
    
    magnitude_valid, magnitude = validate_input(magnitude, 1.0, 10.0)
    if not magnitude_valid:
        st.sidebar.error(f"Magnitude: {magnitude}")
        valid = False

    cdi_valid, cdi = validate_input(cdi, 0, 9)
    if not cdi_valid:
        st.sidebar.error(f"Community Decimal Intensities: {cdi}")
        valid = False

    mmi_valid, mmi = validate_input(mmi, 0, 9)
    if not mmi_valid:
        st.sidebar.error(f"Modified Mercalli Intensity: {mmi}")
        valid = False

    sig_valid, sig = validate_input(sig, 100, 3000)
    if not sig_valid:
        st.sidebar.error(f"Significant Event ID: {sig}")
        valid = False

    dmin_valid, dmin = validate_input(dmin, 1.0, 20.0)
    if not dmin_valid:
        st.sidebar.error(f"Minimum Distance to Station: {dmin}")
        valid = False

    gap_valid, gap = validate_input(gap, 1, 250)
    if not gap_valid:
        st.sidebar.error(f"Azimuthal gap: {gap}")
        valid = False

    depth_valid, depth = validate_input(depth, 1, 1000)
    if not depth_valid:
        st.sidebar.error(f"Depth of Rupture: {depth}")
        valid = False

    if valid:
        input_data = np.array([[magnitude, cdi, mmi, tsunami, sig, dmin, gap, depth]])
        input_data_scaled = minmax_scaler.transform(input_data)
        predicted_cluster = kmeans1.predict(input_data_scaled)[0]
        predicted_cluster_name = cluster_mapping[predicted_cluster]

        st.sidebar.write('\n')
        st.sidebar.header(f"Predicted Cluster:")
        st.sidebar.write(f"{predicted_cluster_name}")


#------------------------------------------------------------------------------------#
