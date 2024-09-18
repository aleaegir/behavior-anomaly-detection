import cv2
import os

def video_to_frames(video_path, output_dir):
    """Converte um vídeo em frames e salva no diretório de saída."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_name, frame)
        frame_count += 1
    
    cap.release()
    print(f'{frame_count} frames extraídos e salvos em {output_dir}')

# Exemplo de uso
video_path = 'data/raw/video.mp4'
output_dir = 'data/processed/frames'
video_to_frames(video_path, output_dir)
