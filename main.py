
from flask import Flask, jsonify, request, Response
from camera import VideoCamera
from flask_cors import CORS
import requests
import re
import cv2
import mediapipe as mp
from wordDetectionLoading.model import callModel

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)

def scrape_signing_savvy(word):
    base_url = 'https://www.signingsavvy.com'
    search_url = f'{base_url}/search/{word}'

    response = requests.get(search_url)

    if response.status_code == 200:
        html_content = response.text

        # Regular expression pattern to extract URLs ending with .mp4
        pattern = r'href="([^"]+\.mp4)"'

        # Using re.search to find the first match
        match = re.search(pattern, html_content)

        if match:
            # Extracting the URL from the match
            mp4_link = match.group(1)
            return mp4_link
        else:
            return "No .mp4 link found"
    else:
        return "Failed to retrieve the webpage"



def extract_frames(video_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    #hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frame_count = 0
    all_hand_landmarks = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8) as hands:
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
                        # landmarks.append({
                        #     'x': landmark.x,
                        #     'y': landmark.y,
                        #     'z': landmark.z
                        # })
                        if landmark.x == None or landmark.y == None:
                            landmarks.extend([0.0,0.0])
                        else:

                            landmarks.extend([landmark.x,landmark.y])

                    if len(landmarks) < 84:
                        remaining = 84 - len(landmarks)
                        landmarks.extend([0.0] * remaining)
                        

                    frame_hand_landmarks.append(landmarks)

                    
                    
                    #if frame_hand_landm


            all_hand_landmarks.append(frame_hand_landmarks)
            frame_count += 1
        if len(all_hand_landmarks) < 200:
            remaining = 200 - len(all_hand_landmarks)
            all_hand_landmarks.extend([[0.0] * 84] * remaining)

    cap.release()
    return all_hand_landmarks




@app.route('/scrape', methods=['GET'])
def scrape():
    word = request.args.get('word', default='', type=str)
    if not word:
        return jsonify({'error': 'No word provided'}), 400

    video_url = scrape_signing_savvy(word)

    if video_url == "No .mp4 link found" or video_url == "Failed to retrieve the webpage":
        return jsonify({'error': video_url}), 404

    return jsonify({'mp4_url': video_url})



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = file.filename
        #file.save(file_path)
        hand_pose_data = extract_frames(file_path)
        prediction = callModel(hand_pose_data)

        print(prediction)


        




        return jsonify({'message': 'File successfully uploaded and frames extracted', 'file_path': file_path, 'predicted_word': prediction}), 200



if __name__ == '__main__':
    app.run(debug=True)
