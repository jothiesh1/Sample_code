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
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

channels = find_usb_cameras()
if not channels:
    print("No USB cameras found.")
    exit()

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

# Constants
PROCESS_EVERY_N_FRAMES = 3
UART_PORT = '/dev/ttyS3'
BAUD_RATE = 9600

# Drowsiness detection parameters
EAR_THRESHOLD = 0.25
DROWSY_FRAMES_THRESHOLD = 20  # Number of frames with low EAR to trigger drowsiness
DROWSY_CONFIRMATION_FRAMES = 5  # Number of recent frames to check for confirmation

# Yawning detection parameters
MAR_THRESHOLD = 0.5
YAWN_FRAMES_THRESHOLD = 7

# Face orientation parameters
ORIENTATION_THRESHOLD = 0.7
ORIENTATION_FRAMES = 15

# Initialize tracking variables
ear_history = deque(maxlen=30)  # Keep track of last 30 EAR values
drowsy_frame_counter = 0
yawn_frame_counter = 0
orientation_counter = 0
frame_counter = 0

# Initialize state flags
drowsy_signal_sent = False
yawn_signal_sent = False

# Landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CENTER_INDICES = [33, 133]
RIGHT_EYE_CENTER_INDICES = [362, 263]
NOSE_TIP_INDEX = 1

def send_serial_data(value, times=1, interval=10):
    try:
        with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {UART_PORT} at {BAUD_RATE} baud")
            ser.flushInput()
            ser.flushOutput()
            for i in range(times):
                byte_data = value.to_bytes(1, byteorder='little')
                ser.write(byte_data)
                print(f"Sent: {value} ({i + 1}/{times})")
                time.sleep(interval)
    except Exception as e:
        print(f"Serial Error: {e}")

def compute_ear(landmarks, eye_indices):
    try:
        # Extract eye points
        eye_points = np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in eye_indices])
        
        # Compute distances
        vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute EAR
        if horizontal_dist == 0:
            return 0
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        
        # Validate EAR value
        if 0.1 <= ear <= 0.5:  # Reasonable EAR range
            return ear
        return 0
    except Exception as e:
        print(f"Error computing EAR: {e}")
        return 0

def compute_mar(landmarks):
    try:
        # Extract mouth points
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_corner = landmarks[78]
        right_corner = landmarks[308]
        
        # Compute distances
        vertical_dist = np.linalg.norm(upper_lip - lower_lip)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        
        # Compute MAR
        if horizontal_dist == 0:
            return 0
        mar = vertical_dist / horizontal_dist
        
        # Validate MAR value
        if 0.1 <= mar <= 1.0:  # Reasonable MAR range
            return mar
        return 0
    except Exception as e:
        print(f"Error computing MAR: {e}")
        return 0

def is_stable_drowsy(ear_history, threshold):
    """Check if drowsiness is stable by looking at recent history"""
    if len(ear_history) < DROWSY_CONFIRMATION_FRAMES:
        return False
    recent_ears = list(ear_history)[-DROWSY_CONFIRMATION_FRAMES:]
    return all(ear < threshold for ear in recent_ears)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
        continue

    start_time = time.time()
    height, width, _ = frame.shape

    # Convert to RGB and process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            landmarks.append([x, y])
        landmarks = np.array(landmarks)

        # Calculate EAR
        left_ear = compute_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = compute_ear(landmarks, RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0

        # Add to history
        if ear > 0:  # Only add valid EAR values
            ear_history.append(ear)

        # Calculate MAR
        mar = compute_mar(landmarks)

        # Display metrics
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Drowsiness detection with improved stability check
        if ear < EAR_THRESHOLD and ear > 0:
            drowsy_frame_counter += 1
            if drowsy_frame_counter >= DROWSY_FRAMES_THRESHOLD:
                if is_stable_drowsy(ear_history, EAR_THRESHOLD) and not drowsy_signal_sent:
                    send_serial_data(25, 2, 10)
                    print(f'Drowsiness detected! EAR: {ear:.3f}')
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    drowsy_signal_sent = True
        else:
            drowsy_frame_counter = max(0, drowsy_frame_counter - 1)
            if drowsy_frame_counter == 0:
                drowsy_signal_sent = False

        # Yawning detection
        if mar > MAR_THRESHOLD:
            yawn_frame_counter += 1
            if yawn_frame_counter >= YAWN_FRAMES_THRESHOLD and not yawn_signal_sent:
                send_serial_data(20, 2, 10)
                print(f'Yawn detected! MAR: {mar:.3f}')
                cv2.putText(frame, "YAWNING DETECTED", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                yawn_signal_sent = True
        else:
            yawn_frame_counter = max(0, yawn_frame_counter - 1)
            if yawn_frame_counter == 0:
                yawn_signal_sent = False

        # Draw facial landmarks
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

    else:
        # Reset counters when no face is detected
        drowsy_frame_counter = 0
        yawn_frame_counter = 0
        drowsy_signal_sent = False
        yawn_signal_sent = False

    # FPS calculation
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display frame
    #cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
