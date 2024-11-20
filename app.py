import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
data = pd.read_csv("modified_jobs.csv")



# Combine relevant text columns for job similarity
data['combined_features'] = data['Job Title'] + ' ' + data['Functional Area'] + ' ' + data['Key Skills'] + ' ' + data['Role Category']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

def recommend_job_titles(input_title, top_n=5):
    # Prepare the input by combining the input title with dummy values for functional area, skills, and role
    input_combined = input_title
    
    # Transform the user's input into the same feature space as the tfidf_matrix
    input_vector = vectorizer.transform([input_combined])
    
    # Compute cosine similarities between input job title and all rows in the dataset
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    
    # Get top N recommendations (excluding the first one which will be the input itself)
    top_indices = similarity_scores.argsort()[0, -top_n-1:-1][::-1]  # Excluding input job title itself
    
    # Return the top N recommendations along with their Job Titles and Similarity Scores
    return data.iloc[top_indices][['Job Title', 'Key Skills', 'Functional Area', 'Role Category']], similarity_scores[0, top_indices]

# Streamlit interface
st.title("Job Recommendation System")

# User input: Either dropdown for job title or text input
input_method = st.radio("Select Input Method", ("Dropdown", "Custom Text"))

if input_method == "Dropdown":
    user_input = st.selectbox("Select Job Title to Find Similar Job Titles", data['Job Title'].unique())
elif input_method == "Custom Text":
    user_input = st.text_input("Enter Job Title to Find Similar Job Titles")

# Ensure user input is not empty
if user_input:
    # Get recommended jobs based on the input
    recommended_jobs, scores = recommend_job_titles(user_input)

    # Display the recommended job titles and their details
    st.subheader("Recommended Job Titles based on Similarity:")
    for idx, row in recommended_jobs.iterrows():
        st.write(f"**Job Title:** {row['Job Title']}")
        st.write(f"**Key Skills:** {row['Key Skills']}")
        st.write(f"**Functional Area:** {row['Functional Area']}")
        st.write(f"**Role Category:** {row['Role Category']}")
        st.write(f"**Similarity Score:** {scores[recommended_jobs.index.get_loc(idx)]}")
        st.write("-" * 40)
else:
    st.warning("Please enter a job title or select one from the dropdown to get recommendations.")
