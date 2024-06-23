
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from camera import VideoCamera
from scrape import scrape_signing_savvy

app = Flask(__name__)
CORS(app)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)