
from flask import Flask, jsonify, request, Response
from camera import VideoCamera
from flask_cors import CORS
import requests
import re

app = Flask(__name__)
CORS(app)

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
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

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


@app.route('/scrape', methods=['GET'])
def scrape():
    word = request.args.get('word', default='', type=str)
    if not word:
        return jsonify({'error': 'No word provided'}), 400

    video_url = scrape_signing_savvy(word)

    if video_url == "No .mp4 link found" or video_url == "Failed to retrieve the webpage":
        return jsonify({'error': video_url}), 404

    return jsonify({'mp4_url': video_url})


if __name__ == '__main__':
    app.run(debug=True)
