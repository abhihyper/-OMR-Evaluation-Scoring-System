"""
OMR Web App (Flask app with modern design)
Features:
- Upload OMR images
- Capture from browser camera
- Server-side webcam snapshot
- Auto-grading with OpenCV
- Eye-catching design with emojis 🎉

Run:
    pip install flask opencv-python imutils numpy pillow
    python omr_web_app.py
"""

from flask import Flask, request, render_template_string, redirect, url_for
import numpy as np
import cv2
import imutils
from imutils import contours as im_contours
import base64

app = Flask(__name__)

# ------------------- OMR Helper Functions -------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

ANSWER_KEY = {0:1, 1:4, 2:0, 3:2, 4:0}

def grade_omr_from_image(image_bgr, answer_key=ANSWER_KEY):
    image = image_bgr.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    if docCnt is None:
        warped_color, warped_gray = image, gray
    else:
        warped_color = four_point_transform(image, docCnt.reshape(4, 2))
        warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            questionCnts.append(c)
    if len(questionCnts) == 0:
        return { 'score': 0, 'total': len(answer_key), 'annotated': warped_color }
    questionCnts = im_contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    annotated = warped_color.copy()
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts_row = im_contours.sort_contours(questionCnts[i:i+5])[0]
        bubbled = None
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j, c)
        k = answer_key.get(q, None)
        if k is not None and bubbled is not None:
            chosen_idx = bubbled[1]
            if chosen_idx == k:
                correct += 1
                cv2.drawContours(annotated, [bubbled[2]], -1, (0, 255, 0), 2)
            else:
                cv2.drawContours(annotated, [bubbled[2]], -1, (0, 0, 255), 2)
                try:
                    correct_cnt = cnts_row[k]
                    cv2.drawContours(annotated, [correct_cnt], -1, (255, 0, 0), 2)
                except Exception:
                    pass
    return { 'score': correct, 'total': len(answer_key), 'annotated': annotated }

def img_to_datauri(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr)
    b64 = base64.b64encode(buf).decode('utf-8')
    return 'data:image/jpeg;base64,' + b64

# ------------------- Website -------------------

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>✨ OMR Grader 🎯</title>
  <style>
    body{ font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); color:#222; padding: 20px; text-align:center; }
    h1{ font-size:2.2em; margin-bottom:8px; }
    h2{ margin-top:0; color:#444 }
    .card{ background:#fff; padding:20px; margin:20px auto; border-radius:16px; box-shadow:0 6px 18px rgba(0,0,0,0.15); max-width:600px; }
    button, input[type=file]{ margin-top:10px; padding:10px 20px; border:none; border-radius:10px; cursor:pointer; background:#66a6ff; color:#fff; font-size:1em; transition:0.3s; }
    button:hover{ background:#4a90e2 }
    video{ width:100%; max-width:400px; border-radius:12px; margin-top:12px; }
    img.result{ max-width:100%; border-radius:12px; margin-top:16px; box-shadow:0 4px 12px rgba(0,0,0,0.1) }
  </style>
</head>
<body>
  <h1>✨ OMR Grader 🎯</h1>
  <h2>Smart MCQ Answer Sheet Checker 📝✅</h2>

  <div class="card">
    <h3>📂 Upload Your OMR Sheet</h3>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <br>
      <button type="submit">🚀 Upload & Grade</button>
    </form>
  </div>

  <div class="card">
    <h3>📸 Capture Using Camera</h3>
    <video id="video" autoplay playsinline></video>
    <br>
    <button id="snap">📷 Capture & Send</button>
    <button id="stop">🛑 Stop Camera</button>
    <div id="result"></div>
  </div>

  <div class="card">
    <h3>💻 Server Webcam (if available)</h3>
    <form action="/capture_server" method="post">
      <button type="submit">🎥 Capture from Server</button>
    </form>
  </div>

<script>
async function startCamera(){
  const video = document.getElementById('video');
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    window._localStream = stream;
  }catch(e){ alert('⚠️ Camera access denied: '+e.message) }
}
function stopCamera(){
  if(window._localStream){ window._localStream.getTracks().forEach(t=>t.stop()); document.getElementById('video').srcObject = null; }
}
startCamera();
document.getElementById('stop').addEventListener('click', stopCamera);
document.getElementById('snap').addEventListener('click', async ()=>{
  const video = document.getElementById('video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg');
  const res = await fetch('/upload_capture', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image: dataUrl }) });
  const html = await res.text();
  document.getElementById('result').innerHTML = html;
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if not f: return redirect(url_for('index'))
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    result = grade_omr_from_image(img)
    datauri = img_to_datauri(result['annotated'])
    return f"<h3>🎯 Score: {result['score']} / {result['total']}</h3><img class='result' src='{datauri}' />"

@app.route('/upload_capture', methods=['POST'])
def upload_capture():
    import json, base64 as b64
    data = request.get_json()
    if not data or 'image' not in data: return 'No image', 400
    img_bytes = b64.b64decode(data['image'].split(',')[1])
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    result = grade_omr_from_image(img)
    datauri = img_to_datauri(result['annotated'])
    return f"<h3>🎯 Score: {result['score']} / {result['total']}</h3><img class='result' src='{datauri}' />"

@app.route('/capture_server', methods=['POST'])
def capture_server():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return '❌ Server camera not available', 500
    ret, frame = cap.read()
    cap.release()
    if not ret: return '❌ Failed to capture', 500
    result = grade_omr_from_image(frame)
    datauri = img_to_datauri(result['annotated'])
    return f"<h3>🎯 Score: {result['score']} / {result['total']}</h3><img class='result' src='{datauri}' />"

if __name__ == '__main__':
    app.run(debug=True)
