from flask import Flask, render_template, Response, jsonify
import cv2
from detector import detection
app = Flask(__name__)

camera = cv2.VideoCapture(0)  


def gen_frames():
    global a
    global detected
    while(True):
        success,frame = camera.read()
        if not success:
            print("Alert ! Camera disconnected")
        else:
            detected,a = detection(success, frame)
            ret,buffer=cv2.imencode('.jpg',detected)
            detected=buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + detected + b'\r\n')
        
 
@app.route('/pred')
def pred():
    return jsonify(result=a)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html', message="{a}".format(a=a))

@app.route('/SLinfo')
def slinfo():
    return render_template('signLanguageInfo.html')

detected=None
a='A'
print(a)
if __name__ == '__main__':
    app.run(debug=True)