from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image

# Load the dataset
df = pd.read_csv("disease_specialist3.csv")
doctors_df = pd.read_csv("doctors.csv", encoding='latin1')
medicine_data = pd.read_csv('final.csv')

X_medicine = medicine_data['drug']
y_disease = medicine_data['disease']

X_train_medicine, X_test_medicine, y_train_disease, y_test_disease = train_test_split(X_medicine, y_disease, test_size=0.2, random_state=42)

medicine_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert medicine names into TF-IDF vectors
    ('clf', MultinomialNB()),  # Use Multinomial Naive Bayes classifier
])

medicine_pipeline.fit(X_train_medicine, y_train_disease)

# Fit the label encoder on the entire dataset
label_encoder = LabelEncoder()
label_encoder.fit(df['Disease'])

# Load the pre-trained model
knn = KNeighborsClassifier(n_neighbors=3)
X = df['Disease'].map(lambda x: label_encoder.transform([x])[0]).values.reshape(-1, 1)
y = df['Specialist']
knn.fit(X, y)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_medicine_name_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(img)

    # Extract medicine names from the extracted text
    # (You'll need to implement this based on the format of the text)
    medicine_names = extract_medicine_names(extracted_text)

    return medicine_names

def extract_medicine_names(text):
    # Placeholder implementation, you'll need to implement this based on your text format
    # Here's a simple example assuming medicine names are separated by newlines
    medicine_names = [name.strip() for name in text.split('\n') if name.strip()]
    return medicine_names

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            medicine_names = extract_medicine_name_from_image(file_path)
            return jsonify({'image': filename, 'medicine_names': medicine_names})
    return jsonify({'error': 'Invalid request'})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if request.method == 'POST':
        medicine_name = request.form['medicine']
        predicted_disease = medicine_pipeline.predict([medicine_name])[0]
        return jsonify({'medicine': medicine_name, 'predicted_disease': predicted_disease})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        disease = request.form['disease']
        disease_encoded = label_encoder.transform([disease])
        specialist = knn.predict(disease_encoded.reshape(-1, 1))[0]
        return jsonify({'disease': disease, 'specialist': specialist})

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        specialty = request.form['specialty']
        doctor_info = doctors_df[doctors_df['specialist'] == specialty]
        if not doctor_info.empty:
            doctor_data = []
            for i in range(1, 4):  # Assuming there are always three doctors listed
                doctor_name = doctor_info.iloc[0][f'{i}st Doctor\'s name']
                contact = doctor_info.iloc[0][f'Contact']
                address = doctor_info.iloc[0][f'Address']
                doctor_data.append({'doctor_name': doctor_name, 'contact': contact, 'address': address})
            return jsonify({'doctors': doctor_data})
        else:
            return jsonify({'error': 'No doctors found for the given specialty.'})



if __name__ == '__main__':
    app.run(debug=True)