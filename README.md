Chest X-ray Analysis Website

This repository contains the code for a Flask-based web application that analyzes chest X-ray images to classify them as Normal or Pneumonia. The application provides an intuitive interface for uploading images, displays detailed classification results, and includes additional features such as an About page and a Contact page.

Features

Image Upload: Users can upload chest X-ray images for classification.

Image Classification: The app uses a trained deep learning model to predict whether the image represents a normal chest X-ray or shows signs of pneumonia.

Detailed Results: The result page provides detailed classification results with probabilities.

About Page: Displays information about the application and its developer.

Contact Page: Allows users to submit feedback or queries, which are stored in an SQLite database.

File Structure

project-directory
│
├── app.py                 # Main Flask application file
├── model.py               # Deep learning model loading and prediction logic
├── static/                # Static files (CSS, images, etc.)
├── templates/             # HTML templates (upload, result, about, contact)
├── database.db            # SQLite database for contact form submissions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── logs/                  # Log files for debugging

Prerequisites

Python 3.8 or higher

Flask

SQLAlchemy

TensorFlow or PyTorch (depending on your model)

SQLite (for contact form storage)

Install the dependencies using:

pip install -r requirements.txt

Usage

Clone the repository:

git clone https://github.com/your-username/chest-xray-analysis.git
cd chest-xray-analysis

Ensure the trained model file is placed in the models/ directory and update the model_path variable in model.py.

Run the application:

python app.py

Open the web application in your browser at http://127.0.0.1:5000.

Application Pages

Home Page: Upload a chest X-ray image for analysis.

Results Page: View the classification result (Normal or Pneumonia) along with details.

About Page: Learn more about the application and the developer.

Contact Page: Submit feedback or queries.

Model Information

The deep learning model used in this application is a Convolutional Neural Network (CNN) trained on a dataset of chest X-ray images. It achieves high accuracy in distinguishing between normal and pneumonia cases.

Logging

Logging is implemented to track application events and errors.

Log files are stored in the logs/ directory.

Future Improvements

Support for additional classifications (e.g., tuberculosis, COVID-19).

Enhanced visualization of model predictions.

Deployment on cloud platforms like AWS or Heroku.

Developer

This project is developed by Piyush Mangalam Bajaj, a final-year Computer Science student passionate about deep learning and its applications in healthcare.

LinkedIn: https://www.linkedin.com/in/piyush-mangalam-bajaj-6a995a236/

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Dataset: Kaggle

Libraries: Flask, TensorFlow/PyTorch, SQLAlchemy

Feel free to contribute to this project by submitting issues or pull requests!

