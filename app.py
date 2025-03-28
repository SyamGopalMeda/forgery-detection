from flask import Flask, request, render_template, send_file, make_response
from io import BytesIO
from PIL import Image
import base64
from detect import detect_forgery

app = Flask(__name__)

# Configuration
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            try:
                # Process the image directly
                img = Image.open(file.stream)
                result_img = detect_forgery(img)
                
                # Convert to bytes
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                result_bytes = BytesIO()
                result_img.save(result_bytes, format='PNG')
                result_bytes.seek(0)
                
                # Create base64 strings for display
                original_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                result_b64 = base64.b64encode(result_bytes.getvalue()).decode('utf-8')
                
                return render_template('result.html',
                                    original=original_b64,
                                    result=result_b64,
                                    original_filename=file.filename,
                                    result_filename=f"result_{file.filename}")
            
            except Exception as e:
                return render_template('upload.html', error=f"Error: {str(e)}")
    
    return render_template('upload.html')

@app.route('/download')
def download():
    img_base64 = request.args.get('img')
    filename = request.args.get('filename', 'result.png')
    img_bytes = base64.b64decode(img_base64)
    
    response = make_response(img_bytes)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', 'attachment', filename=filename)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
