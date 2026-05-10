import math
from src.config import NET_Y_PX

# ── Tuning constants ────────────────────────────────────────────────────
SHOT_COOLDOWN_FRAMES = 35    # ~1.4s at 25fps
MIN_BALL_SPEED       = 6
MIN_TRAJ_LEN         = 8
DIR_CHANGE_THRESHOLD = 0.30  # stricter angle change required

NET_ZONE_PX    = 60    # within 60px of net → smash candidate
VOLLEY_ZONE_PX = 130   # within 130px of net → volley candidate
# ────────────────────────────────────────────────────────────────────────

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PADEL SHOT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FOREHAND  — dominant-hand swing on the right side of body
  BACKHAND  — non-dominant side, arm crosses body midline
  VOLLEY    — intercepts ball before bounce, near the net
  SMASH     — overhead/side overhead near net (Bandeja/Vibora)
  SERVE     — underarm opening shot from behind service line

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class ShotClassifier:

    def __init__(self, fps=25.0, frame_width=1920, frame_height=1080):
        self.fps        = fps
        self.fw         = frame_width
        self.fh         = frame_height
        self.net_y      = NET_Y_PX        # from config — calibrated to camera
        self.last_frame = -SHOT_COOLDOWN_FRAMES
        self.shot_count = 0

    def update(self, frame_idx, timestamp, players, ball_pos, ball_traj):
        if ball_pos is None or not players:
            return None
        if len(ball_traj) < MIN_TRAJ_LEN:
            return None
        if frame_idx - self.last_frame < SHOT_COOLDOWN_FRAMES:
            return None

        bx, by, _ = ball_pos

        speed = self._recent_speed(ball_traj)
        if speed < MIN_BALL_SPEED:
            return None

        if not self._direction_changed(ball_traj):
            return None

        hitter = self._nearest_player(players, bx, by)
        if hitter is None:
            return None

        cx, cy = hitter["center"]
        dist   = math.hypot(cx - bx, cy - by)
        if dist > self.fw * 0.35:
            return None

        shot_type   = self._classify(hitter, players, ball_traj)
        confidence  = self._confidence(ball_traj, speed)
        description = self._describe(shot_type, hitter, speed)

        self.last_frame  = frame_idx
        self.shot_count += 1

        return {
            "frame"      : frame_idx,
            "timestamp"  : round(timestamp, 3),
            "player_id"  : hitter["id"],
            "shot_type"  : shot_type,
            "ball_x"     : bx,
            "ball_y"     : by,
            "ball_speed" : round(speed, 1),
            "confidence" : confidence,
            "description": description,
        }

    def _classify(self, hitter, all_players, traj):
        cx, cy   = hitter["center"]
        net_y    = self.net_y

        mid      = len(traj) // 2
        old_seg  = traj[:mid]
        new_seg  = traj[mid:]

        in_vx    = old_seg[-1][0] - old_seg[0][0]
        in_vy    = old_seg[-1][1] - old_seg[0][1]
        in_speed = math.hypot(in_vx, in_vy)

        out_vx   = new_seg[-1][0] - new_seg[0][0]
        out_vy   = new_seg[-1][1] - new_seg[0][1]
        out_speed = math.hypot(out_vx, out_vy)

        dist_to_net = abs(cy - net_y)

        # ── SMASH ────────────────────────────────────────────────────────
        is_at_net   = dist_to_net < NET_ZONE_PX
        is_powerful = out_speed > 50
        ball_toward = (cy < net_y and in_vy > 0) or (cy > net_y and in_vy < 0)
        if is_at_net and is_powerful and ball_toward:
            return "smash"

        # ── VOLLEY ───────────────────────────────────────────────────────
        if dist_to_net < VOLLEY_ZONE_PX:
            return "volley"

        # ── SERVE ────────────────────────────────────────────────────────
        same_half     = [p for p in all_players
                         if (p["center"][1] > net_y) == (cy > net_y)]
        slow_incoming = in_speed < 10
        if len(same_half) == 1 and slow_incoming:
            return "serve"

        # ── FOREHAND vs BACKHAND ─────────────────────────────────────────
        ball_came_from_right = in_vx < 0
        if cy > net_y:
            return "backhand" if ball_came_from_right else "forehand"
        else:
            return "forehand" if ball_came_from_right else "backhand"

    @staticmethod
    def _recent_speed(traj, window=5):
        if len(traj) < window:
            return 0
        pts   = traj[-window:]
        dists = [math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                 for i in range(1, len(pts))]
        return sum(dists) / len(dists)

    @staticmethod
    def _direction_changed(traj):
        mid    = len(traj) // 2
        old    = traj[:mid]
        new    = traj[mid:]
        vx_old = old[-1][0] - old[0][0]
        vy_old = old[-1][1] - old[0][1]
        vx_new = new[-1][0] - new[0][0]
        vy_new = new[-1][1] - new[0][1]
        mag_old = math.hypot(vx_old, vy_old)
        mag_new = math.hypot(vx_new, vy_new)
        if mag_old < 5 or mag_new < 5:
            return False
        cos_a = (vx_old*vx_new + vy_old*vy_new) / (mag_old * mag_new)
        return cos_a < DIR_CHANGE_THRESHOLD

    @staticmethod
    def _nearest_player(players, bx, by):
        best, best_d = None, float("inf")
        for p in players:
            d = math.hypot(p["center"][0]-bx, p["center"][1]-by)
            if d < best_d:
                best, best_d = p, d
        return best

    @staticmethod
    def _confidence(traj, speed):
        return round(min(len(traj)/MIN_TRAJ_LEN, 1.0)*0.4 + min(speed/40,1.0)*0.6, 2)

    @staticmethod
    def _describe(shot_type, hitter, speed):
        pid = hitter["id"]
        spd = round(speed, 1)
        return {
            "forehand": f"P{pid} forehand drive ({spd}px/f)",
            "backhand": f"P{pid} backhand stroke ({spd}px/f)",
            "volley"  : f"P{pid} net volley ({spd}px/f)",
            "smash"   : f"P{pid} smash near net ({spd}px/f)",
            "serve"   : f"P{pid} underarm serve ({spd}px/f)",
        }.get(shot_type, f"P{pid} shot ({spd}px/f)")
