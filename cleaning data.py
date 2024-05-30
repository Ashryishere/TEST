import pandas as pd
import unicodedata
import re

# Function to clean and normalize text
def clean_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Replace unwanted characters
    text = text.replace('Ã¢Â€Â¢', '-')
    # Remove any other non-alphanumeric characters (optional)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Attempt to read the CSV file with different encodings
def read_csv_with_encoding(file_path):
    try:
        # Try reading with utf-8 encoding
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 doesn't work, try with windows-1252 encoding
        try:
            return pd.read_csv(file_path, encoding='windows-1252')
        except UnicodeDecodeError:
            # If windows-1252 doesn't work, read with utf-8 and ignore errors
            return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

# Read the dataset
file_path = 'C:/Users/Kimo Store/Desktop/AI/UpdatedResumeDataSet.csv'
df = read_csv_with_encoding(file_path)

# Apply the cleaning function to the desired column(s)
df['Resume'] = df['Resume'].apply(clean_text)

# Save the cleaned dataset
df.to_csv('C:/Users/Kimo Store/Desktop/AI/cleaned_dataset.csv', index=False)
