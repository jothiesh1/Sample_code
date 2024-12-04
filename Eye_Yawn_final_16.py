import cv2
import numpy as np
import mediapipe as mp
import time
import serial
import threading
from collections import deque

# Load the YOLOv3-Tiny model for faster detection
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Read class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
phone_class_id = classes.index('cell phone')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def find_usb_cameras():
    channels = []
    for i in range(10):  # Assuming you have at most 10 video devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

channels = find_usb_cameras()
if not channels:
    print("No USB cameras found.")
    
print(f"Detected USB cameras on channels: {channels}")
last_channel = channels[-1]
cap = cv2.VideoCapture(last_channel)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Updated constants for drowsiness detection
EAR_THRESHOLD = 0.3  # Increased from 0.25
DROWSINESS_DURATION = 4  # Increased from 3 seconds
EAR_SMOOTHING_WINDOW = 5  # Window size for EAR smoothing
MIN_EAR_VALUE = 0.15  # Minimum valid EAR value
MAX_EAR_VALUE = 0.45  # Maximum valid EAR value

# Updated constants for yawning detection
MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES = 10
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame

# Constants for face orientation detection
ORIENTATION_RATIO_THRESHOLD = 0.7
ORIENTATION_CONSEC_FRAMES = 15

# Phone detection confidence threshold
PHONE_CONFIDENCE_THRESHOLD = 0.2

# Initialize counters and flags
drowsy_counter = 0
yawn_counter = 0
orientation_counter = 0
frame_counter = 0
valid_ear_count = 0
UART_PORT = '/dev/ttyS3'
BAUD_RATE = 9600

# Initialize flags for sending signals
drowsy_signal_sent = False
yawn_signal_sent = False

# Initialize drowsiness timer and EAR history
drowsy_start_time = None
ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)

# Landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CENTER_INDICES = [33, 133]
RIGHT_EYE_CENTER_INDICES = [362, 263]
NOSE_TIP_INDEX = 1

def send_serial_data(value, times=1, interval=10):
    try:
        with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {UART_PORT} at {BAUD_RATE} baud.")
            ser.flushInput()
            ser.flushOutput()

            for i in range(times):
                byte_data = value.to_bytes(1, byteorder='little')
                ser.write(byte_data)
                print(f"Sent: {value} ({i + 1}/{times})")
                time.sleep(interval)
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

def compute_ear(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    ear = (A + B) / (2.0 * C)
    return ear

def smooth_ear(ear_value, history):
    """Smooth EAR values using moving average"""
    if MIN_EAR_VALUE <= ear_value <= MAX_EAR_VALUE:
        history.append(ear_value)
    if len(history) > 0:
        return sum(history) / len(history)
    return ear_value

def compute_mar_simple(landmarks):
    try:
        p13 = landmarks[13]    # Upper lip
        p14 = landmarks[14]    # Lower lip
        p78 = landmarks[78]    # Left inner lip corner
        p308 = landmarks[308]  # Right inner lip corner
        
        vertical_dist = np.linalg.norm(p13 - p14)
        horizontal_dist = np.linalg.norm(p78 - p308)
        
        mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        return mar
    except Exception as e:
        print(f"Error computing MAR: {e}")
        return 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
        continue

    start_time = time.time()
    height, width, _ = frame.shape

    # Phone detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > PHONE_CONFIDENCE_THRESHOLD and class_id == phone_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Non-max suppression
    if boxes:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, PHONE_CONFIDENCE_THRESHOLD, 0.3)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f"Cell Phone: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Face mesh processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            landmarks.append([x, y])

        landmarks = np.array(landmarks, dtype=np.float64)

        # Calculate and smooth EAR
        left_ear = compute_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = compute_ear(landmarks, RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0
        
        # Apply smoothing and validation
        if MIN_EAR_VALUE <= ear <= MAX_EAR_VALUE:
            smoothed_ear = smooth_ear(ear, ear_history)
            valid_ear_count += 1
        else:
            valid_ear_count = 0
            ear_history.clear()
            smoothed_ear = ear

        # Calculate MAR
        mar = compute_mar_simple(landmarks)

        # Display metrics
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Drowsiness detection with validation
        if valid_ear_count >= EAR_SMOOTHING_WINDOW:
            if smoothed_ear < EAR_THRESHOLD:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                
                drowsy_duration = time.time() - drowsy_start_time
                if drowsy_duration >= DROWSINESS_DURATION and not drowsy_signal_sent:
                    recent_ears = list(ear_history)
                    if all(e < EAR_THRESHOLD for e in recent_ears[-3:]):
                        send_serial_data(25, 2, 10)
                        print(f'Drowsiness detected! EAR: {smoothed_ear:.3f}, Duration: {drowsy_duration:.1f}s')
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 210),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        drowsy_signal_sent = True
            else:
                drowsy_start_time = None
                drowsy_signal_sent = False

        # Yawning detection
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MAR_CONSEC_FRAMES and not yawn_signal_sent:
                send_serial_data(20, 2, 10)
                print('Yawn detected!')
                cv2.putText(frame, "YAWNING DETECTED", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                yawn_signal_sent = True
        else:
            yawn_counter = max(0, yawn_counter - 1)
            if yawn_counter == 0:
                yawn_signal_sent = False

        # Face orientation detection
        left_eye_center = np.mean(landmarks[LEFT_EYE_CENTER_INDICES], axis=0)
        right_eye_center = np.mean(landmarks[RIGHT_EYE_CENTER_INDICES], axis=0)
        nose_tip = landmarks[NOSE_TIP_INDEX]

        dist_left = abs(nose_tip[0] - left_eye_center[0])
        dist_right = abs(nose_tip[0] - right_eye_center[0])
        ratio = dist_left / dist_right if dist_right != 0 else 0

        cv2.putText(frame, f"Orientation Ratio: {ratio:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if abs(1 - ratio) > ORIENTATION_RATIO_THRESHOLD:
            orientation_counter += 1
            if orientation_counter >= ORIENTATION_CONSEC_FRAMES:
                cv2.putText(frame, "FACE ORIENTATION ALERT!", (10, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            orientation_counter = 0

        # Draw facial landmarks
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

    else:
        # Reset all counters when no face is detected
        drowsy_start_time = None
        drowsy_signal_sent = False
        yawn_counter = 0
        yawn_signal_sent = False
        orientation_counter = 0
        valid_ear_count = 0
        ear_history.clear()

    # FPS calculation and display
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display frame
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
