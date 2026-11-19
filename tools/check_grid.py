"""Draw the calibrated road grid (every 5 m along; pavement edges and centre line) onto a frame of any size
with the camera's field of view, to check the calibration by eye.

usage: python3 tools/check_grid.py frame.jpg out.jpg"""
import json, sys
import cv2, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from speed import Road, CFG

cal = CFG['calibration']
road = Road(cal)
img = cv2.imread(sys.argv[1])
sx, sy = img.shape[1] / CFG['frame_size'][0], img.shape[0] / CFG['frame_size'][1]
W = cal['road_width_m']
z0, z1 = cal['zone_u']
th = max(1, int(round(sx)))


def P(u, v):
    x, y = road.to_image(u, v)
    return int(x * sx), int(y * sy)


for v, col in [(0, (0, 255, 255)), (W / 2, (0, 165, 255)), (W, (0, 255, 255))]:
    cv2.polylines(img, [np.array([P(u, v) for u in np.arange(z0 - 4, z1 + 4.01, 0.5)])], False, col, 2 * th)
for u in range(int(np.ceil(z0 / 5) * 5), int(z1) + 1, 5):
    cv2.line(img, P(u, -0.5), P(u, W + 0.5), (255, 0, 255), 2 * th)
    cv2.putText(img, f'{u * 3.2808:.0f} ft', P(u, W + 1.2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sx, (255, 0, 255), th)
for u in (z0, z1):
    cv2.line(img, P(u, -1.5), P(u, W + 1.5), (0, 0, 255), 3 * th)
cv2.imwrite(sys.argv[2], img)
