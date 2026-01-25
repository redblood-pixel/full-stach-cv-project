from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
from PIL import Image
import io
import json
import datetime
from ultralytics import YOLO

import io

from helpers import get_history, create_pdf
from processing import process_image, process_video

app = Flask(__name__)
CORS(app)

# Создаем необходимые директории
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('reports', exist_ok=True)

import requests
import os
from pathlib import Path

def download_ya_disk(public_url, output_name="best.pt"):
    """
    Скачивает файл с публичной ссылки Яндекс.Диска и сохраняет под именем output_name.
    """
    # Шаг 1: Получаем прямую ссылку через Yandex API
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": public_url}
    
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise Exception(f"❌ Ошибка получения ссылки: {response.status_code}, {response.text}")
    
    download_url = response.json().get("href")
    if not download_url:
        raise Exception("❌ Не удалось получить ссылку на скачивание")

    # Шаг 2: Скачиваем файл
    print("🔽 Скачиваю файл...")
    download_response = requests.get(download_url, stream=True)
    download_response.raise_for_status()

    # Определим размер файла для прогресс-бара (опционально)
    total_size = int(download_response.headers.get('content-length', 0))

    with open(output_name, 'wb') as f:
        downloaded = 0
        for chunk in download_response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                done = int(50 * downloaded / total_size)
                percent = downloaded * 100 // total_size
                print(f"\r [{'=' * done}{' ' * (50-done)}] {percent}%", end='', flush=True)
    
    print(f"\n✅ Успешно скачано: {output_name}")


public_link = "https://disk.yandex.ru/d/kiVMroVViIL4gw"  # твоя ссылка
download_ya_disk(public_link, "best.pt")
# Загружаем модель YOLOv8
model = YOLO('best.pt')

# Файл для хранения истории запросов
HISTORY_FILE = 'request_history.json'

# Инициализация файла истории, если он не существует
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)



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
    
    if file.content_type == "video/mp4":
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            video_path = f'static/uploads/{timestamp}_video{os.path.splitext(file.filename)[1]}'
            file.save(video_path)
            return process_video(model, HISTORY_FILE, video_path)
            
        except Exception as e:
            print(str(e))
            return jsonify({'error': str(e)}), 500
    else:

        try:
            # Читаем изображение
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            process_image(model, HISTORY_FILE, image)
            
            return jsonify({
                'success': True
            })
        
        except Exception as e:
            print(str(e))
            return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_request_history():
    """Получение истории запросов"""
    history = get_history(HISTORY_FILE)
    return jsonify(history)

@app.route('/report', methods=['GET'])
def generate_report():
    """Генерация отчета в формате JSON"""
    history = get_history(HISTORY_FILE)
    
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



# Функция для создания PDF отчета для видео анализа
@app.route('/generate-pdf-report', methods=['POST'])
def create_pdf_report():
    """Генерация PDF отчета для видео анализа"""
    try:
        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'error': 'No video ID provided'}), 400
        
        video_id = data['video_id']
        history = get_history(HISTORY_FILE)
        
        # Находим запись видео в истории
        video_entry = None
        for entry in history:
            if entry['id'] == video_id and entry['type'] == 'video_analysis':
                video_entry = entry
                break
        
        if not video_entry:
            return jsonify({'error': 'Video analysis not found'}), 404
        
        # Создаем временный файл для PDF
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f'reports/video_analysis_report_{timestamp}.pdf'
        
        doc = create_pdf(pdf_filename, video_entry)
        
        # Возвращаем PDF файл
        return send_file(pdf_filename, as_attachment=True, mimetype='application/pdf')
        
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)