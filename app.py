import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
data = pd.read_csv("modified_jobs.csv")

# Combine features
data['combined_features'] = data['Job Title'] + ' ' + data['Functional Area'] + ' ' + data['Key Skills'] + ' ' + data['Role Category']

# Vectorize job titles and features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Function to recommend job titles
def recommend_job_titles(input_title, top_n=5):
    input_combined = input_title
    input_vector = vectorizer.transform([input_combined])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0, -top_n-1:-1][::-1]
    return data.iloc[top_indices][['Job Title', 'Key Skills', 'Functional Area', 'Role Category']], similarity_scores[0, top_indices]

# Streamlit interface
st.title("Job Recommendation System")

# Sidebar filters
st.sidebar.header(" Filters")

# Dropdowns for filters
location_options = data['Location'].unique().tolist()
location = st.sidebar.selectbox("🌍 Location", ['All'] + location_options)

qualifications_options = data['Qualifications'].unique().tolist()
qualifications = st.sidebar.selectbox("🎓 Qualifications", ['All'] + qualifications_options)

experience_category_options = data['Experience Category'].unique().tolist()
experience_category = st.sidebar.selectbox("🧑‍💻 Experience Category", ['All'] + experience_category_options)

# User input for job title
user_input = st.text_input("🔎 Enter Job Title to Find Similar Job Titles")

# Search button
if st.button("Search") and user_input:
    # Filter data based on the sidebar inputs
    filtered_data = data.copy()

    if location != 'All':
        filtered_data = filtered_data[filtered_data['Location'] == location]
    if qualifications != 'All':
        filtered_data = filtered_data[filtered_data['Qualifications'] == qualifications]
    if experience_category != 'All':
        filtered_data = filtered_data[filtered_data['Experience Category'] == experience_category]

    # Check if any jobs are left after filtering
    if filtered_data.empty:
        st.warning("⚠️ No jobs found based on the selected filters.")
    else:
        # Recommend jobs
        recommended_jobs, scores = recommend_job_titles(user_input)

        # Filter recommended jobs based on the current filters
        recommended_jobs_filtered = recommended_jobs[recommended_jobs['Job Title'].isin(filtered_data['Job Title'])]

        # Display results in a clean, simple format
        if not recommended_jobs_filtered.empty:
            st.subheader("Recommended Job Titles:")

            # Display each recommendation cleanly with color and emojis
            for idx, row in recommended_jobs_filtered.iterrows():
                st.markdown(f"{row['Job Title']}")
                st.markdown(f"Key Skills: {row['Key Skills']}")
                st.markdown(f"Functional Area: {row['Functional Area']}")
                st.markdown(f"Role Category: {row['Role Category']}")
                st.markdown(f"Similarity Score: {scores[recommended_jobs_filtered.index.get_loc(idx)]:.2f}")
                st.write("-" * 40)

        else:
            st.warning("⚠️ No recommended jobs match the filters.")
# else:
#     st.warning("⚠️ Please enter a job title and click the search button to get recommendations.")
