import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.title("🎬 Movie Recommendation Dashboard (Ages 18–35)")
st.markdown("Explore user behavior, movie clusters, and personalized suggestions with ML-powered insights.")

# Load data with caching
@st.cache_data
def load_data():
    file_path = "full_merged_cleaned_with_clusters.csv"
    if not os.path.exists(file_path):
        st.error(f"The file '{file_path}' was not found.")
        st.stop()

    df = pd.read_csv(file_path)

    # Fix genres: convert string to list safely
    if "genres" in df.columns:
        df["genres"] = df["genres"].astype(str).apply(
            lambda x: [genre.strip() for genre in x.split(',')] if ',' in x else [x]
        )

    return df

# Load and check data
df = load_data()

# Ensure all needed columns are present
required_columns = ["userId", "cluster", "genres", "tag", "title", "rating"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
    st.stop()

# Convert types if needed
df["userId"] = df["userId"].astype(int)
df["cluster"] = df["cluster"].astype(int)

# Sidebar filters
st.sidebar.header("Filters")
clusters = sorted(df["cluster"].dropna().unique())
selected_cluster = st.sidebar.selectbox("Choose a Cluster", clusters)

users_in_cluster = df[df["cluster"] == selected_cluster]["userId"].dropna().unique()
selected_user = st.sidebar.selectbox("Choose a User", users_in_cluster)

# Cluster distribution
st.subheader("👥 User Distribution by Cluster")
cluster_count = df.groupby("cluster")["userId"].nunique().reset_index()
fig1 = px.bar(cluster_count, x="cluster", y="userId",
              labels={"userId": "Number of Users"}, title="Users per Cluster")
st.plotly_chart(fig1)

# Genre preferences by cluster
st.subheader("🎞️ Top Genres in Selected Cluster")
genres_flat = df[df["cluster"] == selected_cluster]["genres"].explode()
top_genres = genres_flat.value_counts().nlargest(10).reset_index()
top_genres.columns = ["Genre", "Count"]
fig2 = px.bar(top_genres, x="Genre", y="Count", title="Top Genres in Cluster")
st.plotly_chart(fig2)

# Top tags overall
st.subheader("🏷️ Top Tags Used")
if "tag" in df.columns:
    top_tags = df["tag"].dropna().value_counts().nlargest(10).reset_index()
    top_tags.columns = ["Tag", "Frequency"]
    fig3 = px.bar(top_tags, x="Tag", y="Frequency", title="Most Frequent Tags")
    st.plotly_chart(fig3)
else:
    st.warning("No 'tag' column found in the dataset.")

# Selected user's movie ratings
st.subheader(f"⭐ Movie Ratings by User {selected_user}")
user_movies = df[df["userId"] == selected_user][["title", "rating"]]
if not user_movies.empty:
    fig4 = px.bar(user_movies, x="title", y="rating", title="Rated Movies",
                  labels={"title": "Movie", "rating": "Rating"})
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4)
else:
    st.warning("This user has no rated movies.")

# Footer
st.markdown("---")
st.markdown("📊 Powered by Clustering, SVD, and Content-Based Filtering")
