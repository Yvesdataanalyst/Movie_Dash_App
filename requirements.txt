import streamlit as st
import pandas as pd
import plotly.express as px
import ast

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("full_merged_cleaned.csv")
    
    # Safely parse genres if needed
    if "genres" in df.columns and isinstance(df["genres"].iloc[0], str):
        try:
            df["genres"] = df["genres"].apply(ast.literal_eval)
        except Exception as e:
            st.error("Error parsing genres: " + str(e))
    return df

df = load_data()

# Check required columns
required_columns = ["userId", "cluster", "genres", "tag", "title", "rating"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
    st.stop()

# Title and intro
st.title(" Movie Recommendation Dashboard (Ages 18–35)")
st.markdown("Explore user behavior, movie clusters, and personalized suggestions with ML-powered insights.")

# Sidebar filters
st.sidebar.header("Filters")
clusters = sorted(df["cluster"].dropna().unique())
selected_cluster = st.sidebar.selectbox("Choose a Cluster", clusters)

users_in_cluster = df[df["cluster"] == selected_cluster]["userId"].dropna().unique()
selected_user = st.sidebar.selectbox("Choose a User", users_in_cluster)

# Cluster distribution
st.subheader(" User Distribution by Cluster")
cluster_count = df.groupby("cluster")["userId"].nunique().reset_index()
fig1 = px.bar(cluster_count, x="cluster", y="userId", labels={"userId": "Number of Users"}, title="Users per Cluster")
st.plotly_chart(fig1)

# Genre preferences by cluster
st.subheader(" Top Genres in Selected Cluster")
genres_flat = df[df["cluster"] == selected_cluster]["genres"].explode()
top_genres = genres_flat.value_counts().nlargest(10).reset_index()
top_genres.columns = ["Genre", "Count"]
fig2 = px.bar(top_genres, x="Genre", y="Count", title="Top Genres in Cluster")
st.plotly_chart(fig2)

# Top tags overall
st.subheader(" Top Tags Used")
if "tag" in df.columns:
    top_tags = df["tag"].dropna().value_counts().nlargest(10).reset_index()
    top_tags.columns = ["Tag", "Frequency"]
    fig3 = px.bar(top_tags, x="Tag", y="Frequency", title="Most Frequent Tags")
    st.plotly_chart(fig3)
else:
    st.warning("No 'tag' column found in the dataset.")

# Selected user's movie ratings
st.subheader(f" Movie Ratings by User {selected_user}")
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
st.markdown(" Powered by Clustering, SVD, and Content-Based Filtering")
