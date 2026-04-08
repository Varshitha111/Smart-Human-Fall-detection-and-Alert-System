import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math
import time
import requests
import smtplib
import os
from datetime import datetime
from email.message import EmailMessage

# ── CONFIG ────────────────────────────────────────────────────────────────

EMAIL_SENDER    = "varshuredd2005@gmail.com"
EMAIL_PASSWORD  = "czcr fbqz inhg efvn"
CARETAKER_EMAIL = "gudavarshithareddy@gmail.com"

NTFY_TOPIC      = "fall_alert"
NTFY_SERVER     = "https://ntfy.sh"

ALERT_COOLDOWN     = 60
last_alert_time    = 0

LOWLIGHT_THRESHOLD = 40
LOWLIGHT_COOLDOWN  = 300
last_lowlight_time = 0

# Stop flag — app.py sets this to False to stop the loop
detection_running  = False

ALERTS_DIR = "alerts"
os.makedirs(ALERTS_DIR, exist_ok=True)

# ── Input source ──────────────────────────────────────────────────────────

# Webcam (default):
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Video file — comment out webcam above and uncomment this:
cap = cv2.VideoCapture(r"C:\Users\varsh\OneDrive\Desktop\major\human-fall-detection\videos\queda.mp4")

if not cap.isOpened():
    print("Error: Cannot open input source")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Input source opened successfully!")

# ── Load MoveNet ──────────────────────────────────────────────────────────

print("Loading MoveNet model...")
model   = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]
print("MoveNet loaded!")

# ── Edges ─────────────────────────────────────────────────────────────────

edges = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

# ── Drawing functions ─────────────────────────────────────────────────────

def draw_keypoints(frame, keypoints, threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (255, 0, 0), -1)


def draw_skeleton(frame, keypoints, edges, threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if c1 > threshold and c2 > threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 2)


def draw_bounding_box(frame, keypoints, threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > threshold:
            x1, y1 = min(x1, kx), min(y1, ky)
            x2, y2 = max(x2, kx), max(y2, ky)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                  (0, 255, 0), 2)


def loop_through_people(frame, keypoints_with_scores, edges, threshold):
    for person in keypoints_with_scores:
        draw_skeleton(frame, person, edges, threshold)
        draw_keypoints(frame, person, threshold)
        draw_bounding_box(frame, person, threshold)

# ── Brightness ────────────────────────────────────────────────────────────

def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray)[0]

# ── Fall detection data ───────────────────────────────────────────────────

y_coordinates = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
x_coordinates = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
aspect_ratios  = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
angles         = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
fall_frames    = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}


def calculate_aspect_ratio(frame, keypoints_with_scores):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    return (x2 - x1) / (y2 - y1 + 1e-6)


def calculate_angle(frame, keypoints_with_scores):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    return math.atan2(centroid_y - y1, centroid_x - x1) * 180 / math.pi


def detect_fall(frame, keypoints_with_scores, frame_count):
    if len(keypoints_with_scores) == 0 or np.all(keypoints_with_scores == 0):
        return False

    for i, person in enumerate(keypoints_with_scores):
        nose_conf = person[0][2]
        if nose_conf < 0.25:
            continue

        angle = calculate_angle(frame, person)
        angles[i + 1].append(angle)

        y_current = person[0][0] * frame.shape[0]
        x_current = person[0][1] * frame.shape[1]
        y_coordinates[i + 1].append(y_current)
        x_coordinates[i + 1].append(x_current)

        if len(y_coordinates[i + 1]) > 5:
            y_diff = y_coordinates[i + 1][-1] - y_coordinates[i + 1][-5]
            x_diff = x_coordinates[i + 1][-1] - x_coordinates[i + 1][-5]
            if x_diff - y_diff > 120 and angle < 30:
                fall_frames[i + 1].append(frame_count)
                return True

        aspect_ratio = calculate_aspect_ratio(frame, person)
        aspect_ratios[i + 1].append(aspect_ratio)

        if aspect_ratio > 2.0 and angle < 30:
            fall_frames[i + 1].append(frame_count)
            return True

    return False

# ── Alert functions ───────────────────────────────────────────────────────

def send_email_alert(snapshot_path=None):
    msg = EmailMessage()
    msg.set_content("URGENT: Fall detected right now! Check immediately.")
    msg['Subject'] = "FALL ALERT - Immediate Action Needed"
    msg['From']    = EMAIL_SENDER
    msg['To']      = CARETAKER_EMAIL

    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, 'rb') as f:
            img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg',
                           filename=os.path.basename(snapshot_path))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent successfully")
    except Exception as e:
        print("Email failed:", e)


def send_lowlight_email(brightness, frame):
    msg = EmailMessage()
    msg.set_content(
        f"WARNING: Very low light detected!\n"
        f"Current brightness: {brightness:.1f} (normal >60-80)\n"
        f"Fall detection may be unreliable until lighting improves.\n"
        f"Snapshot attached for reference."
    )
    msg['Subject'] = "LOW LIGHT WARNING - Fall Detection Camera"
    msg['From']    = EMAIL_SENDER
    msg['To']      = CARETAKER_EMAIL

    _, img_encoded = cv2.imencode('.jpg', frame)
    msg.add_attachment(img_encoded.tobytes(), maintype='image', subtype='jpeg',
                       filename='lowlight_snapshot.jpg')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Low-light warning email sent")
    except Exception as e:
        print("Low-light email failed:", e)


def send_phone_siren_push():
    try:
        r = requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data="URGENT: Fall detected! Check now.".encode(),
            headers={
                "Title":    "FALL ALERT",
                "Priority": "5",
                "Sound":    "siren",
                "Tags":     "warning,rotating_light"
            },
            timeout=10
        )
        print("Siren push sent (status:", r.status_code, ")")
    except Exception as e:
        print("Siren push failed:", e)


def trigger_alerts(frame):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        print("Alert skipped - cooldown active")
        return

    print("Triggering alerts...")
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(ALERTS_DIR, f"fall_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, frame)
    print(f"Snapshot saved: {snapshot_path}")

    send_email_alert(snapshot_path)
    send_phone_siren_push()
    last_alert_time = now

# ── Main detection loop ───────────────────────────────────────────────────

def start_detection():
    global last_alert_time, last_lowlight_time, detection_running
    print("Detection loop starting...")
    prev_time = time.time()

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame or video ended")
            break

        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        brightness = get_brightness(frame)
        if brightness < LOWLIGHT_THRESHOLD:
            cv2.putText(frame, "LOW LIGHT - Visibility Poor", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            now = time.time()
            if now - last_lowlight_time >= LOWLIGHT_COOLDOWN:
                print("Low light detected - sending email warning...")
                send_lowlight_email(brightness, frame)
                last_lowlight_time = now

        img    = frame.copy()
        img    = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 128, 256)
        img    = tf.cast(img, dtype=tf.int32)
        result = movenet(img)
        keypoints_with_scores = result["output_0"].numpy()[:, :, :51].reshape(6, 17, 3)

        loop_through_people(frame, keypoints_with_scores, edges, 0.3)

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if np.any(keypoints_with_scores[:, :, 2] > 0.3):
            if detect_fall(frame, keypoints_with_scores, current_frame):
                cv2.putText(frame, "FALL DETECTED",
                            (frame.shape[1] // 2 - 200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                trigger_alerts(frame)

        cv2.imshow("Fall Detection + Alert System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detection_running = False
    print("Detection loop ended.")

# ── Run directly (for testing only) ──────────────────────────────────────

if __name__ == "__main__":
    detection_running = True
    start_detection()