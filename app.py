import os
import torch
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap4
from models.model import XRayAnalysisModel, load_and_preprocess_image

# Initialize Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contact.db'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize extensions
db = SQLAlchemy(app)
Bootstrap4(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Database model for storing contact details
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)


# Load the trained model with error handling
# Load the trained model with error handling
def load_model():
    model_path = 'best_model_fold0.pth'
    if not os.path.exists(model_path):
        logging.error("Model file not found at path '%s'.", model_path)
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    model = XRayAnalysisModel(num_classes=2)
    try:
        # Load only the weights to avoid any potential security issues
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error("Failed to load model: %s", str(e))
        raise e


# Initialize model at app start-up
try:
    model = load_model()
except Exception as e:
    model = None
    logging.error("Application could not load the model: %s", str(e))


# Prediction function
def predict(image_path):
    logging.info("Starting prediction for image at path '%s'...", image_path)
    try:
        image = load_and_preprocess_image(image_path)
        logging.info("Image tensor shape for prediction: %s", image.shape)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            logging.info("Model prediction: %s", predicted.item())
            return predicted.item()  # Returns class index
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        raise ValueError("Prediction failed due to a model or image processing error.")


# Home route for file upload and prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename != '':
            # Validate the file type
            if uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join('uploads', uploaded_file.filename)
                uploaded_file.save(file_path)

                try:
                    # Perform prediction
                    if model is None:
                        flash("Model could not be loaded. Please contact support.", "danger")
                        return redirect(url_for('upload_file'))

                    prediction = predict(file_path)
                    class_names = ['Normal', 'Pneumonia']
                    result = class_names[prediction]
                    logging.info("Prediction result: %s", result)
                    return render_template('result.html', result=result)
                except ValueError as e:
                    flash(str(e), 'danger')
                except Exception as e:
                    flash('An unexpected error occurred: ' + str(e), 'danger')
            else:
                flash('Please upload a valid image file.', 'danger')

    return render_template('upload.html')


# About page
@app.route('/about')
def about():
    return render_template('about.html')


# Contact page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if name and email and message:
            new_contact = Contact(name=name, email=email, message=message)
            db.session.add(new_contact)
            db.session.commit()
            flash('Thank you for your message!', 'success')
            return redirect(url_for('contact'))
        else:
            flash('Please fill out all fields', 'danger')

    return render_template('contact.html')


# Run the application
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    with app.app_context():
        db.create_all()  # Ensure database tables are created within app context
    app.run(debug=True)