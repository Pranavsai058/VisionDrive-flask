from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ml_model.visualization import process_single_image
import uuid
from pymongo import MongoClient
import gridfs
from datetime import datetime
from bson.objectid import ObjectId

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "visiondrive"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_to_gridfs(file_path, filename, collection_name):
    with open(file_path, 'rb') as file:
        file_id = fs.put(file, filename=filename)
    
    # Store metadata in collection
    collection = db[collection_name]
    doc_id = collection.insert_one({
        'file_id': file_id,
        'filename': filename,
        'upload_date': datetime.utcnow()
    }).inserted_id
    
    return str(doc_id)

def get_file_from_gridfs(file_id, output_path):
    file_data = fs.get(ObjectId(file_id))
    with open(output_path, 'wb') as file:
        file.write(file_data.read())
    return output_path

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if both files were uploaded
        if 'rgb_image' not in request.files or 'lidar_image' not in request.files:
            return redirect(request.url)
        
        rgb_file = request.files['rgb_image']
        lidar_file = request.files['lidar_image']
        
        if rgb_file.filename == '' or lidar_file.filename == '':
            return redirect(request.url)
        
        if rgb_file and allowed_file(rgb_file.filename) and lidar_file and allowed_file(lidar_file.filename):
            # Generate unique ID for this processing session
            session_id = str(uuid.uuid4())
            
            # Save uploaded files locally
            rgb_filename = secure_filename(f"rgb_{session_id}.jpg")
            lidar_filename = secure_filename(f"lidar_{session_id}.jpg")
            
            rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_filename)
            lidar_path = os.path.join(app.config['UPLOAD_FOLDER'], lidar_filename)
            
            rgb_file.save(rgb_path)
            lidar_file.save(lidar_path)
            
            # Save to MongoDB
            rgb_db_id = save_to_gridfs(rgb_path, rgb_filename, "uploads")
            lidar_db_id = save_to_gridfs(lidar_path, lidar_filename, "uploads")
            
            # Process the images
            result_filename = f"result_{session_id}.jpg"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            # Call ML processing function and get direction
            direction = process_single_image(rgb_path, lidar_path, result_path)
            
            # Save result to MongoDB
            result_db_id = save_to_gridfs(result_path, result_filename, "results")
            
            # Store processing record
            processing_record = {
                'session_id': session_id,
                'rgb_file_id': rgb_db_id,
                'lidar_file_id': lidar_db_id,
                'result_file_id': result_db_id,
                'processed_at': datetime.utcnow(),
                'status': 'completed',
                'direction': direction
            }
            db.processing.insert_one(processing_record)
            
            # Pass the result to the template
            return render_template("index.html", 
                                 result_image=url_for('static', filename=f'results/{result_filename}'),
                                 direction=direction)
    
    return render_template("index.html")

@app.route('/history')
def history():
    # Get last 10 processing records
    records = list(db.processing.find().sort('processed_at', -1).limit(10))
    
    # Prepare data for template
    history_data = []
    for record in records:
        # Initialize with None values
        rgb_filename = None
        lidar_filename = None
        result_filename = None
        
        # Safely get RGB file info
        if 'rgb_file_id' in record:
            rgb_file = db.uploads.find_one({'_id': ObjectId(record['rgb_file_id'])})
            if rgb_file and 'filename' in rgb_file:
                rgb_filename = rgb_file['filename']
        
        # Safely get LiDAR file info
        if 'lidar_file_id' in record:
            lidar_file = db.uploads.find_one({'_id': ObjectId(record['lidar_file_id'])})
            if lidar_file and 'filename' in lidar_file:
                lidar_filename = lidar_file['filename']
        
        # Safely get result file info
        if 'result_file_id' in record:
            result_file = db.results.find_one({'_id': ObjectId(record['result_file_id'])})
            if result_file and 'filename' in result_file:
                result_filename = result_file['filename']
        
        history_data.append({
            'session_id': record['session_id'],
            'processed_at': record['processed_at'],
            'rgb_filename': rgb_filename,
            'lidar_filename': lidar_filename,
            'result_filename': result_filename,
            'direction': record.get('direction', 'unknown')
        })
    
    return render_template("history.html", history=history_data)

if __name__ == "__main__":
    app.run(debug=True, port=8000)