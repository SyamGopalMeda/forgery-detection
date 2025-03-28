from flask import Flask, request, render_template, send_from_directory, url_for
import os
from detect import detect_forgery
from datetime import datetime
import glob
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_old_files():
    """Remove files older than 7 days"""
    now = datetime.now()
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
        if os.path.exists(folder):  # Check if folder exists
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (now - file_time).days > 7:
                        os.remove(file_path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    clean_old_files()  # Clean old files on each request
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
            
            # Save original with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = f"original_{timestamp}_{file.filename}"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(original_path)
            
            # Process image - ensure PNG output
            base_name = os.path.splitext(file.filename)[0]
            result_filename = f"result_{timestamp}_{base_name}.png"  # Force PNG extension
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            # Convert and save properly
            detect_forgery(original_path, result_path)
            
            # Generate URLs for template
            original_url = url_for('static', filename=f'uploads/{original_filename}')
            result_url = url_for('static', filename=f'results/{result_filename}')
            
            return render_template('result.html', 
                                original=original_url,
                                original_filename=original_filename,
                                result=result_url,
                                result_filename=result_filename)
    
    return render_template('upload.html')

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)