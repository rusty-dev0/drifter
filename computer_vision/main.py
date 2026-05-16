import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not found")
    exit()

POWERUP_COLOR = "yellow"


colors = {
    "red":    [(0,   120, 70),  (10,  255, 255)],
    "red2":   [(170, 120, 70),  (180, 255, 255)],
    "blue":   [(90,  50,  50),  (130, 255, 255)],
    "green":  [(40,  50,  50),  (80,  255, 255)],
    "yellow": [(20,  100, 100), (30,  255, 255)],
    "orange": [(10,  100, 100), (20,  255, 255)],
    "purple": [(130, 50,  50),  (160, 255, 255)],
}

def get_dominant_color(frame, x, y, w, h):
    pad_x = w // 4
    pad_y = h // 4
    roi = frame[y + pad_y: y + h - pad_y, x + pad_x: x + w - pad_x]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_label = None
    best_count = 0

    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_label = color_name

    if best_label == "red2":
        best_label = "red"
    return best_label

def draw_star(img, cx, cy, size, color):
    points = []
    for i in range(10):
        angle = np.radians(i * 36 - 90)
        r = size if i % 2 == 0 else size // 2
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        points.append([px, py])
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1)

smoothed = {}
SMOOTH_FRAMES = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()

    raw_detected = {}

    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            dominant = get_dominant_color(frame, x, y, w, h)
            if dominant is None:
                continue

            cx = x + w // 2
            cy = y + h // 2

            if dominant not in raw_detected:
                raw_detected[dominant] = []
            raw_detected[dominant].append((cx, cy, w, h))

    merged_detected = {}
    for label, blobs in raw_detected.items():
        merged = []
        used = [False] * len(blobs)
        for i, (cx1, cy1, w1, h1) in enumerate(blobs):
            if used[i]:
                continue
            gx, gy, gw, gh = cx1, cy1, w1, h1
            count = 1
            for j, (cx2, cy2, w2, h2) in enumerate(blobs):
                if i == j or used[j]:
                    continue
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
                if dist < 120:
                    gx += cx2; gy += cy2
                    gw = max(gw, w2); gh = max(gh, h2)
                    count += 1
                    used[j] = True
            used[i] = True
            merged.append((gx // count, gy // count, gw, gh))
        merged_detected[label] = merged

    for label in list(smoothed.keys()):
        if label not in merged_detected:
            del smoothed[label]

    for label, blobs in merged_detected.items():
        if label not in smoothed:
            smoothed[label] = []
        smoothed[label].append(blobs)
        if len(smoothed[label]) > SMOOTH_FRAMES:
            smoothed[label].pop(0)

    final_detected = []
    for label, history in smoothed.items():
        if not history or not history[-1]:
            continue
        latest = history[-1]
        for (cx, cy, w, h) in latest:
            avg_cx = int(np.mean([f[0][0] for f in history if f]))
            avg_cy = int(np.mean([f[0][1] for f in history if f]))
            final_detected.append((label, avg_cx, avg_cy, w, h))

    for label, cx, cy, w, h in final_detected:
        x = cx - w // 2
        y = cy - h // 2

        if label == POWERUP_COLOR:
            draw_star(output, cx, cy, 30, (0, 215, 255))
            cv2.putText(output, "POWERUP", (cx - w//2, cy - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        else:
            cv2.rectangle(output, (x, y), (x+w, y+h), (200, 200, 200), 2)
            cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
    parts = [str(len(final_detected))]
    for label, cx, cy, w, h in final_detected:
        obj = f"{label} {cx} {cy} {w} {h}"
        parts.append(obj)
    output_bytes = b"|".join(p.encode() for p in parts)
    print(output_bytes)

    cv2.imshow("Color Detection", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()