from flask import Flask, render_template, request, redirect, Response, send_file
import flask
from flask.helpers import flash
import cv2,imutils,time
import pyshine as ps
import numpy as np
import os
import urllib.request

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

UPLOAD_FOLDER = "libs/insight_face/InsightFace_Pytorch/data/facebank/current_face"

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW
cwd = os.getcwd()


from detreg import detection 
app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
   return render_template('index.html')

def pyshine_process(params):
    print("Parameters:",params)
    """Video streaming generator function."""
    CAMERA=False
    if CAMERA:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("https://demo.bahien.com/live/stream/playlist.m3u8")
        # cap = cv2.VideoCapture("rtsp://rtmp.bahien.com:1935/live/live")
    print('FUNCTION DONE')
    # Read until video is completed
    fps=0
    st=0
    frames_to_count=20
    cnt=0


    while cap.isOpened():

        ret, img = cap.read()
        if ret == True:
            START_TIME = time.time()
            if cnt == frames_to_count:
                try: # To avoid divide by 0 we put it in try except
                    fps = round(frames_to_count/(time.time()-st))
                    st = time.time()
                    cnt=0
                except:
                    pass
            
            cnt = cnt + 1
            img, cropface = detection(img)
            img = imutils.resize(img, width=1280)

            text  =  'FPS: '+str(fps)
            img = ps.putBText(img,text,text_offset_x=20,text_offset_y=30,background_RGB=(10,20,222))
            frame = cv2.imencode('.JPEG', img,[cv2.IMWRITE_JPEG_QUALITY,100])[1].tobytes()
            # time.sleep(0.016)
            print("time cost: ", time.time() - START_TIME)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cv2.imwrite("/static/1.jpeg",cropface)
        else:
            pass


@app.route('/res',methods = ['POST','GET'])
def res():
    global result
    if request.method == 'POST':
        result = request.form.to_dict()
        return render_template("index.html",result = result)

@app.route('/results')
def video_feed1():
    global result
    params = result
    return Response(pyshine_process(params),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def video_feed2():
    return send_file("static/1.jpg", mimetype='image/gif')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		for file in files:
			if file and allowed_file(file.filename):
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
		flash('File(s) successfully uploaded')
		return redirect('/')

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",threaded=True)