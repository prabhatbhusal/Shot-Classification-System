import numpy as np

# ── Court boundary polygon (pixel coordinates for 1920x1080) ────────────
# Trapezoid shape — narrow at top (far side), wide at bottom (near side)
# Extended bottom to y=800 to capture near-camera players
#
#   [370,30]─────────────────[1130,30]     far side (north)
#       \                          /
#        \        NET              /
#         \                      /
#   [0,800]────────────────[1920,800]      near side (south)
#
COURT_POLYGON = np.array([
    [370,   30],    # top-left
    [1130,  30],    # top-right
    [1920, 800],    # bottom-right  ← extended wide + low
    [0,    800],    # bottom-left   ← extended wide + low
], dtype=np.int32)

# Net Y pixel position in this camera view
NET_Y_PX = 200    # net appears at roughly y=200 from top

# Ball filtering
FILTER_BALL_TO_COURT = True
