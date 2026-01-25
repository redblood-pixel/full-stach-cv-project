import datetime
import os
from flask import jsonify
import cv2
from PIL import Image

from src.utils.helpers import draw_detections

def process_image(app_context, image):
    try:
        model = app_context.get_model()
        uploads_dir = app_context.get_uploads_dir()
        results_dir = app_context.get_results_dir()

        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        timestamp = app_context.get_timestamp()
        image_path = os.path.join(uploads_dir, f'{timestamp}_original.jpg')
        image.save(image_path, 'JPEG', quality=95)

        results = model(image)
        
        result_image = draw_detections(image, results)
        
        result_path = os.path.join(results_dir, f'{timestamp}_result.jpg')
        
        if result_image.mode == 'RGBA':
            result_image = result_image.convert('RGB')
        
        result_image.save(result_path, 'JPEG', quality=95)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
        
        history = app_context.get_history()
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'image_upload',
            'original_image': image_path,
            'result_image': result_path,
            'detections': detections,
            'summary': {d['class']: sum(1 for d in detections if d['class'] == d['class']) for d in detections}
        }
        history.append(history_entry)
        app_context.save_history(history)

        return {'success': True}

    except Exception as e:
        print(str(e))
        return {'error': str(e)}, 500

def process_video(app_context, video_path: str):
    try:
        model = app_context.get_model()
        uploads_dir = app_context.get_uploads_dir()
        results_dir = app_context.get_results_dir()
        frame_interval_seconds = app_context.get_video_frame_interval()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = int(fps * frame_interval_seconds)
        
        frame_results = []
        all_detections = []
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                results = model(image)
                
                result_image = draw_detections(image, results)
                
                result_timestamp = app_context.get_timestamp()
                result_path = os.path.join(results_dir, f'{result_timestamp}_video_frame.jpg')
                
                if result_image.mode == 'RGBA':
                    result_image = result_image.convert('RGB')
                
                result_image.save(result_path, 'JPEG', quality=95)
                
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        detection_data = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': box.xyxy[0].tolist()
                        }
                        detections.append(detection_data)
                        all_detections.append(detection_data)
                
                frame_results.append({
                    'frame_number': frame_count,
                    'timestamp': result_timestamp,
                    'result_image': result_path,
                    'detections': detections
                })
            
            frame_count += 1
        
        cap.release()
        
        summary = {}
        for detection in all_detections:
            class_name = detection['class']
            if class_name not in summary:
                summary[class_name] = 0
            summary[class_name] += 1
        
        history = app_context.get_history()
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'video_analysis',
            'original_video': video_path,
            'frame_results': frame_results,
            'summary': summary
        }
        history.append(history_entry)
        app_context.save_history(history)
        
        return jsonify({
            'success': True,
            'frame_results': frame_results,
            'summary': summary
        })
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500