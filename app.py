from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import sys
import json
import datetime
from PIL import Image
import io

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from src.config.config import Config
from src.services.application import ApplicationContext
from src.utils.helpers import create_pdf
from src.services.processing import process_image, process_video

# Create configuration and application context
cfg = Config()
app_context = ApplicationContext(cfg)

# Initialize the application context
app_context.initialize()

# Create Flask app
app = Flask(__name__)
CORS(app, origins=cfg.get_cors_origins())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        if file.content_type.startswith('video/'):
            timestamp = app_context.get_timestamp()
            video_path = os.path.join(
                app_context.get_uploads_dir(),
                f'{timestamp}_video{os.path.splitext(file.filename)[1]}'
            )
            file.save(video_path)
            return process_video(app_context, video_path)
        else:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            process_image(app_context, image)
            
            return jsonify({'success': True})
            
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_request_history():
    history = app_context.get_history()
    return jsonify(history)

@app.route('/report', methods=['GET'])
def generate_report():
    history = app_context.get_history()
    
    stats = {
        'total_requests': len(history),
        'total_detections': 0,
        'timeline': {}
    }
    
    for entry in history:
        if 'detections' in entry:
            stats['total_detections'] += len(entry['detections'])
        elif 'frame_results' in entry:
            for frame in entry['frame_results']:
                stats['total_detections'] += len(frame['detections'])
        
        date = entry['timestamp'][:10]
        if date not in stats['timeline']:
            stats['timeline'][date] = 0
        stats['timeline'][date] += 1
    
    report = {
        'generated_at': datetime.datetime.now().isoformat(),
        'statistics': stats,
        'recent_entries': history[-10:]
    }
    
    report_filename = os.path.join(
        app_context.get_reports_dir(),
        f'report_{app_context.get_timestamp()}.json'
    )
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return send_file(report_filename, as_attachment=True)

@app.route('/generate-pdf-report', methods=['POST'])
def create_pdf_report():
    try:
        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'error': 'No video ID provided'}), 400
        
        video_id = data['video_id']
        history = app_context.get_history()
        
        video_entry = None
        for entry in history:
            if entry['id'] == video_id and entry['type'] == 'video_analysis':
                video_entry = entry
                break
        
        if not video_entry:
            return jsonify({'error': 'Video analysis not found'}), 404
        
        pdf_filename = os.path.join(
            app_context.get_reports_dir(),
            f'video_analysis_report_{app_context.get_timestamp()}.pdf'
        )
        
        create_pdf(pdf_filename, video_entry)
        
        return send_file(pdf_filename, as_attachment=True, mimetype='application/pdf')
        
    except Exception as e:
        app.logger.error(f"Error generating PDF report: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        debug=cfg.get_app_debug(),
        host=cfg.get_app_host(),
        port=cfg.get_app_port()
    )