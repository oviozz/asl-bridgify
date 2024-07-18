
import cv2
import numpy as np
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from camera import VideoCamera
from scrape import scrape_signing_savvy
from wordDetectionLoading.model import callModel
import mediapipe as mp
import os

from RAG import answer_query_with_rag

app = Flask(__name__)
CORS(app)

video_camera = None

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
@app.route('/video_feed')
def video_feed():
    global video_camera
    video_camera = VideoCamera()
    return Response(video_camera.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global video_camera
    if video_camera:
        video_camera.stop_stream()
    return jsonify({'message': 'Stopping the video feed'})


def extract_frames(video_path):
    # Initialize MediaPipe Hands and Drawing
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # Create a Hands object with the desired parameters
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    frame_count = 0
    all_hand_landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    if landmark.x is None or landmark.y is None:
                        landmarks.extend([0.0, 0.0])
                    else:
                        landmarks.extend([landmark.x, landmark.y])
                if len(landmarks) < 84:
                    remaining = 84 - len(landmarks)
                    landmarks.extend([0.0] * remaining)
                elif len(landmarks) > 84:
                    landmarks = landmarks[:84]
                frame_hand_landmarks.extend(landmarks)
            if len(frame_hand_landmarks) < 84:
                frame_hand_landmarks.extend([0.0] * (84 - len(frame_hand_landmarks)))
            elif len(frame_hand_landmarks) > 84:
                frame_hand_landmarks = frame_hand_landmarks[:84]
        else:
            frame_hand_landmarks = [0.0] * 84
        all_hand_landmarks.append(frame_hand_landmarks)
        frame_count += 1
    # Pad with empty frames if fewer than 200 frames
    while len(all_hand_landmarks) < 200:
        all_hand_landmarks.append([0.0] * 84)
    # Truncate if more than 200 frames
    all_hand_landmarks = all_hand_landmarks[:200]
    cap.release()
    final_landmarks = np.array(all_hand_landmarks)
    print(final_landmarks.shape)  # Should print (200, 84)
    return final_landmarks

@app.route('/scrape', methods=['GET'])
def scrape():
    word = request.args.get('word', default='', type=str)

    if not word:
        return jsonify({'error': 'No word provided'}), 400

    video_url = scrape_signing_savvy(word)

    if video_url == "No .mp4 link found" or video_url == "Failed to retrieve the webpage":
        return jsonify({'error': video_url}), 404

    return jsonify({'mp4_url': video_url})

@app.route("/uploader", methods=['POST'])
def upload_file():
    process_video_dir = 'processVideoFile'
    ensure_directory(process_video_dir)

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(process_video_dir, file.filename)
        file.save(file_path)

        hand_pose_data = extract_frames(file_path)
        prediction = callModel(hand_pose_data)

        return jsonify({
            'message': 'File successfully uploaded and frames extracted',
            'file_path': file_path,
            'predicted_word': prediction
        }), 200

    return jsonify({'error': 'Unexpected error occurred'}), 500

@app.route('/callRAG', methods=['GET'])
def callRAG():
    user_query = request.args.get('query')
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    system_instructions = "Please provide a detailed and comprehensive answer with actionable advice for improving ASL skills based on this user's profile: Include tips for improving letter, word, and sentence hand gesture formation, and emphasize the importance of facial expressions and body language."
    
    response = answer_query_with_rag(user_query, system_instructions)

    return jsonify({"response": response.content})



# @app.route('/calllearningPlanRAG', methods=['GET'])
# def calllearningPlanRAG():
#     user_query = request.args.get('query')
#     if not user_query:
#         return jsonify({"error": "Query parameter is required"}), 400

#     system_instructions = "You are an american sign language assistant that provides structured learning plans for learning letters,words, and sentences"
    
#     response = answer_query_with_rag("generate me a beginner friendly american sign language learning roadmap", system_instructions)

#     return jsonify({"response": response.content})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)