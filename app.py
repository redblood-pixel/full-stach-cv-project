from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import io
import json
import datetime
from ultralytics import YOLO
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

# Создаем необходимые директории
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Загружаем модель YOLOv8
model = YOLO('yolov8n.pt')

# Файл для хранения истории запросов
HISTORY_FILE = 'request_history.json'

# Инициализация файла истории, если он не существует
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

def get_history():
    """Получение истории запросов из файла"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history(history):
    """Сохранение истории запросов в файл"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def draw_detections(image, results):
    """Отрисовка результатов детекции на изображении (только игрушки)"""
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # COCO dataset class index for teddy bear
    TEDDY_BEAR_CLASS = 77
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получаем класс и уверенность
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Пропускаем, если это не игрушка (плюшевый мишка)
            if cls != TEDDY_BEAR_CLASS:
                continue
            
            # Получаем имя класса
            class_name = model.names[cls]
            
            # Получаем координаты бокса
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Рисуем подпись
            label = f'{class_name} {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Читаем изображение
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Преобразуем RGBA в RGB, если необходимо
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Сохраняем оригинальное изображение
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        original_path = f'static/uploads/{timestamp}_original.jpg'
        image.save(original_path, 'JPEG', quality=95)
        
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
                
                # Пропускаем, если это не игрушка (плюшевый мишка)
                if cls != TEDDY_BEAR_CLASS:
                    continue
                
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Сохраняем запись в историю
        history = get_history()
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'image_upload',
            'original_image': original_path,
            'result_image': result_path,
            'detections': detections,
            'summary': {d['class']: sum(1 for d in detections if d['class'] == d['class']) for d in detections}
        }
        history.append(history_entry)
        save_history(history)
        
        return jsonify({
            'success': True,
            'result_image': result_path,
            'detections': detections,
            'summary': history_entry['summary']
        })
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_request_history():
    """Получение истории запросов"""
    history = get_history()
    return jsonify(history)

@app.route('/report', methods=['GET'])
def generate_report():
    """Генерация отчета в формате JSON"""
    history = get_history()
    
    # Подсчет статистики
    stats = {
        'total_requests': len(history),
        'total_detections': 0,
        'class_distribution': {},
        'timeline': {}
    }
    
    for entry in history:
        # Подсчет общего количества детекций
        entry_detections = len(entry['detections'])
        stats['total_detections'] += entry_detections
        
        # Распределение по классам
        for class_name, count in entry['summary'].items():
            if class_name not in stats['class_distribution']:
                stats['class_distribution'][class_name] = 0
            stats['class_distribution'][class_name] += count
        
        # Таймлайн по датам
        date = entry['timestamp'][:10]  # YYYY-MM-DD
        if date not in stats['timeline']:
            stats['timeline'][date] = 0
        stats['timeline'][date] += 1
    
    # Создаем отчет
    report = {
        'generated_at': datetime.datetime.now().isoformat(),
        'statistics': stats,
        'recent_entries': history[-10:]  # Последние 10 записей
    }
    
    # Сохраняем отчет
    report_filename = f'reports/report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return send_file(report_filename, as_attachment=True)

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """Анализ видео файла на наличие игрушек"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Сохраняем загруженное видео
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        video_path = f'static/uploads/{timestamp}_video{os.path.splitext(file.filename)[1]}'
        file.save(video_path)
        
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
                        
                        # Пропускаем, если это не игрушка (плюшевый мишка)
                        if cls != TEDDY_BEAR_CLASS:
                            continue
                        
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
        history = get_history()
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'video_analysis',
            'original_video': video_path,
            'frame_results': frame_results,
            'summary': summary
        }
        history.append(history_entry)
        save_history(history)
        
        return jsonify({
            'success': True,
            'frame_results': frame_results,
            'summary': summary
        })
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

# Функция для создания PDF отчета
@app.route('/generate-pdf-report', methods=['POST'])
def create_pdf_report():
    """Генерация PDF отчета для группы изображений"""
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'error': 'No results provided'}), 400
        
        batch_results = data['results']
        
        # Создаем временный файл для PDF
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f'reports/toy_detection_report_{timestamp}.pdf'
        
        # Создаем документ
        doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))
        elements = []
        styles = getSampleStyleSheet()
        
        # Заголовок отчета
        title = Paragraph("Toys detection report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Подзаголовок с датой
        date_str = datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        subtitle = Paragraph(f"Report generation date: {date_str}", styles['Normal'])
        elements.append(subtitle)
        elements.append(Spacer(1, 24))
        
        # Статистика по всем изображениям
        total_images = len(batch_results)
        total_detections = 0
        detections_per_image = []
        
        for result in batch_results:
            detections_count = len(result['result']['detections'])
            total_detections += detections_count
            detections_per_image.append(detections_count)
        
        # Создаем таблицу со статистикой
        stats_data = [
            ['Metric', 'Values'],
            ['Total images', str(total_images)],
            ['Total toys count', str(total_detections)],
            ['Avarage total count', f"{total_detections/total_images:.2f}" if total_images > 0 else '0'],
            ['Max toys count', str(max(detections_per_image) if detections_per_image else 0)],
            ['Min toys count', str(min(detections_per_image) if detections_per_image else 0)]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 24))
        
        # Создаем график с matplotlib
        plt.figure(figsize=(10, 5))
        plt.plot(detections_per_image, marker='o', linewidth=2, markersize=6)
        plt.title('Detected toys counted')
        plt.xlabel('Image number')
        plt.ylabel('Toys Count')
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график в байты
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Добавляем график в PDF
        img = ReportLabImage(img_buffer, width=500, height=250)
        elements.append(img)
        elements.append(Spacer(1, 24))
        
        # Таблица с результатами по каждому изображению
        table_data = [['Nu.', 'File Name', 'Toys Count']]
        for i, result in enumerate(batch_results, 1):
            filename = result['filename']
            detections_count = len(result['result']['detections'])
            table_data.append([str(i), filename, str(detections_count)])
        
        # Ограничиваем количество строк в таблице для первой страницы
        max_rows_first_page = 15
        table_data_first_page = table_data[:max_rows_first_page + 1]
        
        result_table = Table(table_data_first_page)
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(result_table)
        
        # Добавляем текст о дополнительных страницах
        if len(table_data) > max_rows_first_page + 1:
            elements.append(Spacer(1, 12))
            more_pages_text = Paragraph(f"И полные результаты для оставшихся {len(table_data) - max_rows_first_page - 1} изображений.", styles['Normal'])
            elements.append(more_pages_text)
        
        # Собираем документ
        doc.build(elements)
        
        # Возвращаем PDF файл
        return send_file(pdf_filename, as_attachment=True, mimetype='application/pdf')
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)