import cv2
import numpy as np
from src.config import COURT_POLYGON, NET_Y_PX

# ── Colour palette (BGR) ─────────────────────────────────────────────────
SHOT_COLOURS = {
    "forehand" : (0,   210, 255),   # amber
    "backhand" : (255,  80, 180),   # magenta
    "volley"   : (0,   255, 140),   # spring green
    "smash"    : (0,    60, 255),   # red
    "serve"    : (255, 200,   0),   # sky blue
}
PLAYER_COLOURS = {
    1: (57,  255,  20),   # neon green
    2: (0,   191, 255),   # sky blue
    3: (255, 165,   0),   # orange
    4: (180,   0, 255),   # purple
}
DEFAULT_COL  = (200, 200, 200)
FONT         = cv2.FONT_HERSHEY_DUPLEX
FONT_SM      = cv2.FONT_HERSHEY_SIMPLEX
SHOT_FLASH   = 22
# ─────────────────────────────────────────────────────────────────────────


class Visualizer:
    def __init__(self, width, height):
        self.w          = width
        self.h          = height
        self._last_shot = None
        self._flash     = 0
        self._shot_log  = []

    def draw(self, frame, players, ball, trajectory, shot, ball_tracker=None):
        self._draw_court_zone(frame)
        self._draw_net_line(frame)
        self._draw_trajectory(frame, trajectory)
        self._draw_players(frame, players)
        self._draw_ball(frame, ball, ball_tracker)
        self._draw_shot_burst(frame, shot)
        self._draw_hud(frame, players, ball, ball_tracker)
        self._draw_shot_log(frame)
        return frame

    # ── court + net ──────────────────────────────────────────────────────

    @staticmethod
    def _draw_court_zone(frame):
        overlay = frame.copy()
        cv2.polylines(overlay, [COURT_POLYGON], True, (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    @staticmethod
    def _draw_net_line(frame):
        """Draw a dashed net line across the frame at NET_Y_PX."""
        h, w = frame.shape[:2]
        y    = NET_Y_PX
        # Dashed line
        dash, gap = 30, 15
        x = 0
        while x < w:
            cv2.line(frame, (x, y), (min(x+dash, w), y), (200, 200, 0), 1)
            x += dash + gap

    # ── players ──────────────────────────────────────────────────────────

    @staticmethod
    def _draw_players(frame, players):
        for p in players:
            pid    = p["id"]
            col    = PLAYER_COLOURS.get(pid, DEFAULT_COL)
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cx     = int(p["center"][0])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

            # Corner bracket accents
            L = 14
            for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                                    (x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (sx, sy), (sx+dx*L, sy), col, 3)
                cv2.line(frame, (sx, sy), (sx, sy+dy*L), col, 3)

            # Label pill above box
            label = f"P{pid}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            px1, py1 = x1, max(y1 - th - 10, 0)
            px2, py2 = x1 + tw + 10, y1
            cv2.rectangle(frame, (px1, py1), (px2, py2), col, -1)
            cv2.putText(frame, label, (px1+5, py2-4),
                        FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Dot at feet
            cv2.circle(frame, (cx, y2), 4, col, -1)

    # ── ball ─────────────────────────────────────────────────────────────

    def _draw_ball(self, frame, ball, ball_tracker=None):
        if ball is None:
            return
        x, y, r = ball

        # Check if this is a Kalman prediction or real detection
        is_pred = ball_tracker is not None and ball_tracker.is_predicted

        if is_pred:
            # Predicted position — dashed ring in dimmer colour
            cv2.circle(frame, (x, y), r + 8, (0, 180, 180), 1)
            cv2.circle(frame, (x, y), r + 2, (0, 180, 180), 1)
            cv2.putText(frame, "~", (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,180,180), 1)
        else:
            # Real detection — bright cyan rings
            cv2.circle(frame, (x, y), r + 6, (0, 255, 255), 1)   # outer glow
            cv2.circle(frame, (x, y), r + 2, (0, 255, 255), 2)   # main ring
            cv2.circle(frame, (x, y), 3,     (255, 255, 255), -1) # core

    # ── trajectory ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_trajectory(frame, traj):
        if len(traj) < 2:
            return
        n = len(traj)
        for i in range(1, n):
            alpha = i / n
            col   = (0, int(180*alpha), int(255*alpha))
            thick = max(1, int(alpha * 3))
            cv2.line(frame,
                     (int(traj[i-1][0]), int(traj[i-1][1])),
                     (int(traj[i][0]),   int(traj[i][1])),
                     col, thick)
        # Arrow at tip
        if len(traj) >= 4:
            cv2.arrowedLine(frame,
                            (int(traj[-4][0]), int(traj[-4][1])),
                            (int(traj[-1][0]), int(traj[-1][1])),
                            (0, 255, 255), 2, tipLength=0.5)

    # ── shot burst ───────────────────────────────────────────────────────

    def _draw_shot_burst(self, frame, shot):
        if shot is not None:
            self._last_shot = shot
            self._flash     = SHOT_FLASH
            self._shot_log.append(shot)
            if len(self._shot_log) > 7:
                self._shot_log.pop(0)

        if self._last_shot is None or self._flash <= 0:
            return
        self._flash -= 1

        s   = self._last_shot
        col = SHOT_COLOURS.get(s["shot_type"], DEFAULT_COL)
        bx, by = int(s["ball_x"]), int(s["ball_y"])

        # Expanding pulse ring
        pulse_r = int(18 + (SHOT_FLASH - self._flash) * 3)
        alpha   = self._flash / SHOT_FLASH
        ov      = frame.copy()
        cv2.circle(ov, (bx, by), pulse_r, col, 3)
        cv2.addWeighted(ov, alpha*0.7, frame, 1-alpha*0.7, 0, frame)

        # Badge
        label = s["shot_type"].upper()
        bx_c  = max(130, min(bx, self.w - 130))
        by_c  = max(70, by - 35)
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.0, 2)
        pad   = 8
        r1    = (bx_c - tw//2 - pad, by_c - th - pad*2)
        r2    = (bx_c + tw//2 + pad, by_c + pad)
        cv2.rectangle(frame, (r1[0]-2, r1[1]-2), (r2[0]+2, r2[1]+2),
                      (255,255,255), -1)
        cv2.rectangle(frame, r1, r2, col, -1)
        cv2.putText(frame, label, (r1[0]+pad, r2[1]-pad),
                    FONT, 1.0, (0,0,0), 2, cv2.LINE_AA)

        # Sub info
        info = f"P{s['player_id']}  {int(s['confidence']*100)}% conf"
        cv2.putText(frame, info, (r1[0], r2[1]+18),
                    FONT_SM, 0.5, col, 1, cv2.LINE_AA)

        # Connector line to ball
        cv2.line(frame, (bx_c, r2[1]), (bx, by), col, 1, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, players, ball, ball_tracker=None):
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (235, 95), (15,15,15), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

        cv2.rectangle(frame, (0,0), (235,22), (40,40,40), -1)
        cv2.putText(frame, "SHOT CLASSIFICATION SYSTEM",
                    (6,16), FONT_SM, 0.42, (0,200,255), 1, cv2.LINE_AA)

        is_pred  = ball_tracker is not None and ball_tracker.is_predicted
        if ball:
            ball_str = f"({ball[0]}, {ball[1]})" + (" ~pred" if is_pred else "")
        else:
            ball_str = "searching..."

        lines = [
            f"Players : {len(players)}",
            f"Ball    : {ball_str}",
            f"Shots   : {len(self._shot_log)}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (8, 38+i*18),
                        FONT_SM, 0.48, (220,220,220), 1, cv2.LINE_AA)

        for pid, col in PLAYER_COLOURS.items():
            lx = 8 + (pid-1) * 57
            cv2.rectangle(frame, (lx, 79), (lx+48, 91), col, -1)
            cv2.putText(frame, f"P{pid}", (lx+16, 90),
                        FONT_SM, 0.38, (0,0,0), 1, cv2.LINE_AA)

    # ── shot log ─────────────────────────────────────────────────────────

    def _draw_shot_log(self, frame):
        if not self._shot_log:
            return
        log_w = 205
        log_h = len(self._shot_log) * 28 + 32
        lx    = self.w - log_w - 8
        ly    = 8

        ov = frame.copy()
        cv2.rectangle(ov, (lx, ly), (lx+log_w, ly+log_h), (15,15,15), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "RECENT SHOTS",
                    (lx+8, ly+18), FONT_SM, 0.42, (0,200,255), 1, cv2.LINE_AA)

        for i, s in enumerate(reversed(self._shot_log)):
            col   = SHOT_COLOURS.get(s["shot_type"], DEFAULT_COL)
            row_y = ly + 30 + i * 28
            cv2.circle(frame, (lx+14, row_y+6), 6, col, -1)
            txt = f"P{s['player_id']}  {s['shot_type'][:3].upper()}  {s['timestamp']:.1f}s"
            cv2.putText(frame, txt, (lx+26, row_y+12),
                        FONT_SM, 0.44, (220,220,220), 1, cv2.LINE_AA)
