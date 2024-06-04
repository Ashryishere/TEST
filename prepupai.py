import os
import logging
from flask import Flask, request, jsonify
import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import zipfile
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    with zipfile.ZipFile('LastModelRFC.zip', 'r') as zip_ref:
        zip_ref.extractall('')
    with open('LastModelRFC.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('LastVECTOR.pkl', 'rb') as file:
        loaded_vectorizer = pickle.load(file)
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

# Function to calculate similarity scores
def calculate_similarity_scores(vectorizer, job_descs, resume_text):
    try:
        resume_vectorized = vectorizer.transform([resume_text])
        scores = []

        for job_desc in job_descs:
            job_desc_vectorized = vectorizer.transform([job_desc])
            score = cosine_similarity(job_desc_vectorized, resume_vectorized)
            scores.append(score[0][0])

        return scores
    except Exception as e:
        logging.error(f"Error calculating similarity scores: {e}")
        return None

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file
        cv_path = 'uploaded_cv.pdf'
        file.save(cv_path)

        # Extract text from uploaded CV
        resume_text = extract_text_from_pdf(cv_path)
        if resume_text is None:
            return jsonify({'error': 'Failed to extract text from the PDF'}), 500

        # Provide the full path to the CSV file
        csv_file_path = os.getenv('CSV_FILE_PATH', 'job_descriptions.csv')

        # Check if the CSV file exists
        if not os.path.isfile(csv_file_path):
            return jsonify({'error': 'Job descriptions CSV file not found'}), 404

        # Load job descriptions
        job_desc_data = pd.read_csv(csv_file_path)
        job_descriptions = job_desc_data['Job Description']

        # Calculate similarity scores
        similarity_scores = calculate_similarity_scores(loaded_vectorizer, job_descriptions, resume_text)
        if similarity_scores is None:
            return jsonify({'error': 'Failed to calculate similarity scores'}), 500

        # Normalize the similarity scores to percentages
        normalized_scores = [score * 100 for score in similarity_scores]

        # Get top 3 results with normalized scores
        top_results = sorted(zip(job_desc_data['Job Title'], normalized_scores), key=lambda x: x[1], reverse=True)[:3]

        # Create a bar plot
        titles = [result[0] for result in top_results]
        scores = [result[1] for result in top_results]
        plt.bar(titles, scores)
        plt.xlabel('Job Title')
        plt.ylabel('Similarity Score (%)')
        plt.title('Top 3 Matching Job Positions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot image
        plot_image_dir = os.getenv('PLOT_IMAGE_DIR', '')
        if not os.path.exists(plot_image_dir):
            os.makedirs(plot_image_dir)
        plot_image_path = os.path.join(plot_image_dir, 'results.png')
        plt.savefig(plot_image_path)

        plt.close()  # Close the plot to free resources

        # Check if the image file exists
        if not os.path.exists(plot_image_path):
            return jsonify({'error': 'Failed to save the plot image'}), 500

        # Open the saved image automatically
        img = Image.open(plot_image_path)
        img.show()

        # Delete the uploaded CV after processing
        os.remove(cv_path)

        return jsonify({'results': top_results, 'plot_image': plot_image_path})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
