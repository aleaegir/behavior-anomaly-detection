from flask import Flask, render_template, request
import cv2
from Scripts.anomaly_detection import detect_anomalies, load_anomaly_model

app = Flask(__name__)

model = load_anomaly_model('models/saved_models/anomaly_detector.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return 'Nenhum arquivo de vídeo fornecido.', 400

    video = request.files['video']
    video_path = f'data/raw/{video.filename}'
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    anomalies = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if detect_anomalies(model, frame):
            anomalies.append(frame_count)
        frame_count += 1

    cap.release()
    return render_template('results.html', anomalies=anomalies, total_frames=frame_count)

if __name__ == '__main__':
    app.run(debug=True)
