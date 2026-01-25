import datetime
from flask import jsonify
import cv2
from PIL import Image

from helpers import get_history, save_history, draw_detections

def process_image(model, history_file, image):

    try:

         # Преобразуем RGBA в RGB, если необходимо
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Сохраняем оригинальное изображение
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = f'static/uploads/{timestamp}_original.jpg'
        image.save(image_path, 'JPEG', quality=95)

         # Выполняем детекцию
        results = model(image)
        
        # Отрисовываем результаты
        result_image = draw_detections(image, results)
        
        # Сохраняем результат
        result_path = f'static/results/{timestamp}_result.jpg'
        
        # Преобразуем RGBA в RGB, если необходимо
        if result_image.mode == 'RGBA':
            result_image = result_image.convert('RGB')
        
        result_image.save(result_path, 'JPEG', quality=95)
        
        # Подготавливаем данные для ответа
        detections = []
        TEDDY_BEAR_CLASS = 77  # COCO dataset class index for teddy bear
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                print(cls)
                print(results)
                
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Сохраняем запись в историю
        history = get_history(history_file)
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
        save_history(history_file, history)

    except Exception as e:
        print(str(e))
        return {'error': str(e)}, 500

def process_video(model, history_file, video_path: str):
    """Анализ видео файла на наличие игрушек"""
    
    try:
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 500
        
        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Определяем интервал извлечения кадров (каждые 2 секунды)
        frame_interval = int(fps * 2)
        
        frame_results = []
        all_detections = []
        
        frame_count = 0
        extracted_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обрабатываем кадр с определенным интервалом
            if frame_count % frame_interval == 0:
                # Конвертируем BGR в RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Конвертируем в PIL изображение
                image = Image.fromarray(rgb_frame)
                
                # Выполняем детекцию
                results = model(image)
                
                # Отрисовываем результаты
                result_image = draw_detections(image, results)
                
                # Сохраняем результат
                result_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                result_path = f'static/results/{result_timestamp}_video_frame.jpg'
                
                # Преобразуем RGBA в RGB, если необходимо
                if result_image.mode == 'RGBA':
                    result_image = result_image.convert('RGB')
                
                result_image.save(result_path, 'JPEG', quality=95)
                
                # Подготавливаем данные для ответа
                detections = []
                TEDDY_BEAR_CLASS = 77  # COCO dataset class index for teddy bear
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        print(cls)
                        
                        class_name = model.names[cls]
                        
                        detection_data = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': box.xyxy[0].tolist()
                        }
                        detections.append(detection_data)
                        all_detections.append(detection_data)
                
                # Добавляем результат кадра
                frame_results.append({
                    'frame_number': frame_count,
                    'timestamp': result_timestamp,
                    'result_image': result_path,
                    'detections': detections
                })
                
                extracted_frame_count += 1
                
            frame_count += 1
        
        # Освобождаем ресурсы
        cap.release()
        
        # Создаем сводку
        summary = {}
        for detection in all_detections:
            class_name = detection['class']
            if class_name not in summary:
                summary[class_name] = 0
            summary[class_name] += 1
        
        # Сохраняем запись в историю
        history = get_history(history_file)
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'video_analysis',
            'original_video': video_path,
            'frame_results': frame_results,
            'summary': summary
        }
        history.append(history_entry)
        save_history(history_file, history)
        
        return jsonify({
            'success': True,
            'frame_results': frame_results,
            'summary': summary
        })
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500