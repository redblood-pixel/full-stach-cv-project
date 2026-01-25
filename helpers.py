import json
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import datetime
import io

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet


def get_history(file):
    """Получение истории запросов из файла"""
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history(file, history):
    """Сохранение истории запросов в файл"""
    with open(file, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def draw_detections(image, results):
    """Отрисовка результатов детекции на изображении (только игрушки)"""
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получаем класс и уверенность
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            print(cls)
            
            # Получаем координаты бокса
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    
    return Image.fromarray(img)

def create_pdf(pdf_filename, video_entry):

    # Создаем документ
        doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))
        elements = []
        styles = getSampleStyleSheet()
        
        # Заголовок отчета
        title = Paragraph("Video Analysis Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Подзаголовок с датой
        date_str = datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        subtitle = Paragraph(f"Report generation date: {date_str}", styles['Normal'])
        elements.append(subtitle)
        elements.append(Spacer(1, 24))
        
        # Информация о видео
        video_info = [
            ['Video ID', f"{video_entry['id']}"] ,
            ['Analysis Date', f"{video_entry['timestamp'][:19].replace('T', ' ')}"] ,
            ['Total Frames Processed', f"{len(video_entry['frame_results'])}"]
        ]
        
        info_table = Table(video_info)
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 24))
        
        # Статистика по детекциям
        total_detections = sum(video_entry['summary'].values())
        
        stats_data = [
            ['Metric', 'Values'],
            ['Total Detections', str(total_detections)],
            ['Unique Objects', str(len(video_entry['summary']))]
        ]
        
        # Добавляем информацию о каждом объекте
        for class_name, count in video_entry['summary'].items():
            stats_data.append([f"{class_name} Count", str(count)])
        
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
        
        # Подготавливаем данные для графика
        frame_numbers = [frame['frame_number'] for frame in video_entry['frame_results']]
        detections_count = [len(frame['detections']) for frame in video_entry['frame_results']]
        
        plt.plot(frame_numbers, detections_count, marker='o', linewidth=2, markersize=6)
        plt.title('Detections per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Detections')
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
        
        # Добавляем примеры кадров с детекциями
        elements.append(Paragraph("Sample Frames with Detections:", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Добавляем до 6 кадров (или меньше, если их меньше)
        sample_frames = video_entry['frame_results'][:6]
        for frame in sample_frames:
            # Добавляем информацию о кадре
            frame_info = f"Frame {frame['frame_number']} - {len(frame['detections'])} detections"
            elements.append(Paragraph(frame_info, styles['Normal']))
            
            # Добавляем изображение кадра
            frame_img = ReportLabImage(frame['result_image'], width=300, height=200)
            elements.append(frame_img)
            elements.append(Spacer(1, 12))
        
        # Собираем документ
        doc.build(elements)