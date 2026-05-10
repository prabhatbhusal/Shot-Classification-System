import cv2
import numpy as np
from ultralytics import YOLO
from src.config import COURT_POLYGON, FILTER_BALL_TO_COURT, NET_Y_PX


class PlayerTracker:
    """
    Detects and tracks players using YOLOv8.
    Uses PURE POSITION-BASED ID assignment every frame — no stale ID map.
    
    ID assignment logic (top-down camera):
        Top half of court    → P1 (left), P2 (right)
        Bottom half of court → P3 (left), P4 (right)
    """

    def __init__(self, model_path="yolov8x.pt", device=0):
        print("[INFO] Loading YOLOv8x model (downloads on first run ~130MB)...")
        self.model  = YOLO(model_path)
        self.device = device
        print("[INFO] YOLO model ready!")

    def detect(self, frame):
        results = self.model.track(
            source  = frame,
            persist = True,
            classes = [0],         # person only
            device  = self.device,
            verbose = False,
            conf    = 0.15,        # low threshold — catch partially visible players
            iou     = 0.4,
        )

        boxes = results[0].boxes
        if boxes is None:
            return []

        # Collect all valid detections inside court
        raw = []
        ids = boxes.id  # may be None if tracking lost
        for i, box in enumerate(boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2
            # Use feet (bottom center) for court boundary check
            feet_y = min(float(y2), frame.shape[0] - 1)

            if not _inside_court(cx, feet_y):
                continue

            raw.append({
                "bbox"  : [float(x1), float(y1), float(x2), float(y2)],
                "center": (float(cx), float(cy)),
            })

        if not raw:
            return []

        # ── Position-based stable ID assignment ──────────────────────────
        # Split into top half (far side) and bottom half (near side)
        net_y   = NET_Y_PX
        top     = sorted([p for p in raw if p["center"][1] < net_y],
                         key=lambda p: p["center"][0])   # sort by X
        bottom  = sorted([p for p in raw if p["center"][1] >= net_y],
                         key=lambda p: p["center"][0])   # sort by X

        players = []
        for i, p in enumerate(top[:2]):
            p["id"] = i + 1   # P1=left, P2=right
            players.append(p)
        for i, p in enumerate(bottom[:2]):
            p["id"] = i + 3   # P3=left, P4=right
            players.append(p)

        return players


class BallTracker:
    """
    Improved ball tracker using three layered techniques:

    1. Background Subtraction (MOG2)
       Since the camera is completely fixed (CCTV mount), the court surface
       is always static. MOG2 learns the background and flags anything moving
       as foreground — the ball will always appear here when in motion.

    2. HSV Color Masking
       Cross-references the MOG2 foreground with the ball's expected color
       (yellow/neon-green for padel balls, white as fallback). Only candidates
       that appear in BOTH masks are accepted — massively reduces false positives.

    3. Kalman Filter
       Predicts where the ball SHOULD be when detection fails (occlusion,
       motion blur, bounce). Provides smooth sub-frame position estimates
       so the trajectory never jumps wildly.

    Detection priority each frame:
        a) Try MOG2 + HSV intersection  → most reliable
        b) Try HSV alone                → slower ball / after bounce
        c) Use Kalman prediction        → ball momentarily hidden
    """

    TRAJ_LEN  = 25
    MAX_MISS  = 12    # frames before trajectory is cleared
    MAX_JUMP  = 250   # pixels — larger jump = false positive

    def __init__(self):
        self.trajectory  = []
        self._miss_count = 0
        self._predicted  = False   # True when using Kalman prediction

        # ── Background subtractor ────────────────────────────────────────
        # history=300: learns background over ~12 seconds at 25fps
        # varThreshold=40: higher = less sensitive to subtle changes
        # detectShadows=False: don't waste time classifying shadows
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history        = 300,
            varThreshold   = 40,
            detectShadows  = False,
        )

        # ── Kalman filter ────────────────────────────────────────────────
        # State  : [x, y, vx, vy]  — position + velocity
        # Measure: [x, y]          — only position is observed
        self._kf = cv2.KalmanFilter(4, 2)

        # Transition matrix: x' = x + vx,  y' = y + vy
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we observe x and y only
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Noise covariances — tuned for a fast-moving small ball
        self._kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self._kf.errorCovPost        = np.eye(4, dtype=np.float32)

        self._kalman_ready = False   # True once we have a first detection

    # ── public API (unchanged interface) ─────────────────────────────────

    def detect(self, frame):
        """
        Returns (x, y, radius) of the ball if found or predicted.
        Returns None only when completely lost.
        Updates self.trajectory.
        """
        # Always run background subtractor (it needs every frame to learn)
        fg_mask  = self._bg.apply(frame)
        hsv_mask = self._build_hsv_mask(frame)

        ball = None

        # ── Method 1: MOG2 foreground ∩ HSV color ────────────────────────
        combined = cv2.bitwise_and(fg_mask, hsv_mask)
        ball     = self._find_ball_in_mask(combined, min_area=4, max_area=500)

        # ── Method 2: HSV alone (catches slow / just-bounced ball) ────────
        if ball is None:
            ball = self._find_ball_in_mask(hsv_mask, min_area=4, max_area=350)

        # ── Court boundary filter ─────────────────────────────────────────
        if ball and FILTER_BALL_TO_COURT:
            bx, by, _ = ball
            if not _inside_court(bx, by):
                ball = None

        # ── Jump sanity check ─────────────────────────────────────────────
        if ball and self.trajectory:
            bx, by, _ = ball
            lx, ly    = self.trajectory[-1]
            if ((bx-lx)**2 + (by-ly)**2)**0.5 > self.MAX_JUMP:
                ball = None   # reject teleporting detection

        # ── Kalman update / predict ───────────────────────────────────────
        if ball:
            bx, by, r    = ball
            self._predicted = False
            self._miss_count = 0

            measurement = np.array([[np.float32(bx)],
                                     [np.float32(by)]])
            if not self._kalman_ready:
                # Initialise Kalman state on first detection
                self._kf.statePre = np.array(
                    [[np.float32(bx)], [np.float32(by)],
                     [np.float32(0)],  [np.float32(0)]]
                )
                self._kalman_ready = True

            self._kf.correct(measurement)
            self._kf.predict()

        else:
            self._miss_count += 1

            # Use Kalman prediction if we had a recent detection
            if self._kalman_ready and self._miss_count <= self.MAX_MISS // 2:
                pred = self._kf.predict()
                px   = int(pred[0][0])
                py   = int(pred[1][0])
                # Only use prediction if it's inside court
                if _inside_court(px, py):
                    ball            = (px, py, 4)   # radius=4 for predicted ball
                    self._predicted = True
                else:
                    self._predicted = False

            # Clear stale trajectory after too many misses
            if self._miss_count >= self.MAX_MISS:
                self.trajectory      = []
                self._kalman_ready   = False
                self._predicted      = False

        # ── Update trajectory ─────────────────────────────────────────────
        if ball:
            bx, by, _ = ball
            self.trajectory.append((bx, by))
            if len(self.trajectory) > self.TRAJ_LEN:
                self.trajectory.pop(0)

        return ball

    def get_velocity(self, window=4):
        if len(self.trajectory) < window:
            return None
        p0 = self.trajectory[-window]
        p1 = self.trajectory[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    @property
    def is_predicted(self):
        """True when current position is a Kalman prediction, not a real detection."""
        return self._predicted

    # ── private ───────────────────────────────────────────────────────────

    def _build_hsv_mask(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Padel ball: neon yellow-green
        mask_y = cv2.inRange(hsv,
                              np.array([20,  80,  80]),
                              np.array([40, 255, 255]))
        # White ball (some padel balls are white)
        mask_w = cv2.inRange(hsv,
                              np.array([0,   0,  200]),
                              np.array([180, 30, 255]))

        mask   = cv2.bitwise_or(mask_y, mask_w)
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)
        return mask

    @staticmethod
    def _find_ball_in_mask(mask, min_area=4, max_area=500):
        """
        Finds the best ball candidate in a binary mask.
        Scores each contour by circularity and closeness to expected ball size.
        Returns (x, y, radius) or None.
        """
        contours, _ = cv2.findContours(mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best       = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue

            circularity = 4 * np.pi * area / (perim ** 2)
            if circularity < 0.35:
                continue

            # Score: prefer circular shapes near expected ball area (~20px²)
            score = circularity * (1 / (abs(area - 20) / 20 + 1))

            if score > best_score:
                best_score = score
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                best = (int(x), int(y), max(int(radius), 3))

        return best


# ── Shared utility ───────────────────────────────────────────────────────
def _inside_court(cx, cy):
    """Returns True if point is inside the court polygon."""
    result = cv2.pointPolygonTest(
        COURT_POLYGON,
        (float(cx), float(cy)),
        measureDist=False
    )
    return result >= 0
