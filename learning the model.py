import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your local job descriptions dataset
local_dataset_path = r'C:/Users/Kimo Store/Desktop/AI/final Dataset.csv'
job_desc_data = pd.read_csv(local_dataset_path)

# Replace NaN values with an empty string in 'Job Description' column
job_desc_data['Job Description'] = job_desc_data['Job Description'].fillna('')

# Check for NaN values in 'Job Title' and replace them if necessary
job_desc_data['Job Title'] = job_desc_data['Job Title'].fillna('Unknown')

# Load the resume dataset
resume_data = pd.read_csv(r'C:/Users/Kimo Store/Desktop/AI/cleaned_dataset.csv')

# Vectorize the text data
tfidf_vector = TfidfVectorizer(sublinear_tf=True, stop_words='english')
tfidf_vector.fit(job_desc_data['Job Description'].values)  # Fit the vectorizer on the job descriptions

# Transform the job descriptions into feature vectors
X = tfidf_vector.transform(job_desc_data['Job Description'].values)
Y = job_desc_data['Job Title'].values

# Ensure there are no NaN values in the target variable
if pd.isnull(Y).any():
    raise ValueError("The target variable (Y) contains NaN values.")

# Train RandomForest Classifier
model_RF = RandomForestClassifier(n_estimators=200)
Model_RFC = model_RF.fit(X, Y)

# Save the trained model and vectorizer
pickle_dir = r'C:\Users\Kimo Store\Desktop\pkl'
os.makedirs(pickle_dir, exist_ok=True)  # Create the directory if it doesn't exist

with open(os.path.join(pickle_dir, 'LastModelRFC.pkl'), 'wb') as file:
    pickle.dump(Model_RFC, file)

with open(os.path.join(pickle_dir, 'LastVectorizer.pkl'), 'wb') as file:
    pickle.dump(tfidf_vector, file)
