# object-detection-app
from flask import Flask, Response, render_template
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialise webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

def generate_frames():
    """ Generate video frames with object detection."""
    while True:
        success, frame = cap.read()
        if not success:
            break

  # Resize frame for faster processing
   frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape

 # Prepare frame for YOLO
  blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

  # Process detections
   boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

   # Apply non-max suppression
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

   # Draw bounding boxes and labels
   for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 255, 0)  # Green for bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

   # Encode frame as JPEG
   ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

  # Stream frame as MJPEG
  yield (b'--frame\r\n'
  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main webpage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video frames with detected objects."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

