"""Fit the camera-image -> road-plane model from points picked on a camera frame (tools/picker.html).

The model is a one-term radial lens correction plus a homography: image pixel -> (u, v) in metres, where
u runs along the road (0 at your reference landmark, positive away from the camera) and v runs across it
(0 at the near pavement edge, road_width at the far edge).

Constraints used:
  * landmarks:   points with a measured distance along the road (u). Only u is used, so posts, mailboxes,
                 driveway edges etc. can sit off the pavement. Give v too ("v": metres) if you know it.
  * near_edge / far_edge / centre_line: points clicked along those lines (v = 0, width, width / 2).
    Click several points along each line; the polyline between them is sampled densely.
At least 3 landmarks spread along the road plus both edges are needed; 4-6 landmarks and the centre line help.

usage: python3 tools/calibrate.py points.json config.json [--no-lens]
Updates config.json in place (calibration + road_polygon, scaled to frame_size) and writes calib_check.jpg
next to points.json if the image is found there."""
import json, os, sys
import numpy as np

FT = 0.3048

pts_path, cfg_path = sys.argv[1], sys.argv[2]
fit_lens = '--no-lens' not in sys.argv
P = json.load(open(pts_path))
cfg = json.load(open(cfg_path))
scale = FT if P.get('units', 'ft') == 'ft' else 1.0
Wimg, Himg = P['image_size']
W = float(P['road_width']) * scale
fw, fh = cfg['frame_size']

landmarks = P.get('landmarks', [])
if len(landmarks) < 3 or len(P.get('near_edge', [])) < 2 or len(P.get('far_edge', [])) < 2:
    sys.exit('need >= 3 landmarks and >= 2 points on each of near_edge and far_edge')


def densify(poly, step=15.0):
    """Points every `step` px along a clicked polyline."""
    poly = np.asarray(poly, float)
    out = [poly[0]]
    for a, b in zip(poly, poly[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


near, far = densify(P['near_edge']), densify(P['far_edge'])
centre = densify(P['centre_line']) if len(P.get('centre_line', [])) >= 2 else np.zeros((0, 2))
lm = np.array([l['px'] for l in landmarks], float)
lm_u = np.array([l['u'] for l in landmarks], float) * scale
lm_v = [(i, l['v'] * scale) for i, l in enumerate(landmarks) if l.get('v') is not None]

CX, CY, RN, S = Wimg / 2, Himg / 2, Wimg / 2, 1000.0


def undistort(pts, k1):
    d = (pts - [CX, CY]) / RN
    return [CX, CY] + d * (1 + k1 * (d ** 2).sum(1, keepdims=True)) * RN


def apply(h, pts):
    H = np.append(h[:8], 1).reshape(3, 3)
    p = np.c_[undistort(pts, h[8]) / S, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def residuals(h, near, far, centre):
    r = [3.0 * (apply(h, lm)[:, 0] - lm_u), apply(h, near)[:, 1], apply(h, far)[:, 1] - W]
    if len(centre):
        r.append(apply(h, centre)[:, 1] - W / 2)
    if lm_v:
        idx = [i for i, _ in lm_v]
        r.append(3.0 * (apply(h, lm[idx])[:, 1] - [v for _, v in lm_v]))
    return np.concatenate(r)


def nearest_on(arr, p):
    return arr[np.argmin(((arr - p) ** 2).sum(1))]


# initial homography (no lens term): the two most separated landmarks, projected sideways onto both edges
i0, i1 = np.argmin(lm_u), np.argmax(lm_u)
src = np.array([nearest_on(near, lm[i0]), nearest_on(near, lm[i1]), nearest_on(far, lm[i0]), nearest_on(far, lm[i1])]) / S
dst = np.array([[lm_u[i0], 0], [lm_u[i1], 0], [lm_u[i0], W], [lm_u[i1], W]])
A = []
for (x, y), (u, v) in zip(src, dst):
    A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
H0 = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
h = np.append((H0 / H0[2, 2]).ravel()[:8], 0.0)


def fit(h, near, far, centre):
    n = 9 if fit_lens else 8
    lam = 1e-3
    for _ in range(400):
        r0 = residuals(h, near, far, centre)
        J = np.empty((r0.size, n))
        for k in range(n):
            d = np.zeros(9); d[k] = 1e-6 * max(1, abs(h[k]))
            J[:, k] = (residuals(h + d, near, far, centre) - r0) / d[k]
        M = J.T @ J
        step = np.zeros(9)
        step[:n] = np.linalg.solve(M + lam * np.diag(np.diag(M)), -J.T @ r0)
        if (residuals(h + step, near, far, centre) ** 2).sum() < (r0 ** 2).sum():
            h, lam = h + step, lam / 3
            if np.abs(step).max() < 1e-10:
                break
        else:
            lam *= 5
    return h


h = fit(h, near, far, centre)
for _ in range(2):  # drop edge samples more than 0.5 m off (mis-clicks), refit
    near = near[np.abs(apply(h, near)[:, 1]) < 0.5]
    far = far[np.abs(apply(h, far)[:, 1] - W) < 0.5]
    if len(centre):
        centre = centre[np.abs(apply(h, centre)[:, 1] - W / 2) < 0.5]
    h = fit(h, near, far, centre)

unit = P.get('units', 'ft')
print('landmarks: measured vs fitted distance along the road')
for l, (uf, vf) in zip(landmarks, apply(h, lm)):
    print(f"  {l.get('name', ''):16s} {l['u']:8.1f} {unit}   fitted {uf / scale:8.1f} {unit}   (error {(uf / scale - l['u']):+.2f})")
for name, arr, t in (('near edge', near, 0), ('far edge', far, W), ('centre line', centre, W / 2)):
    if len(arr):
        e = apply(h, arr)[:, 1] - t
        print(f'  {name:12s} rms {np.sqrt((e ** 2).mean()) / scale:.2f} {unit}, max {np.abs(e).max() / scale:.2f} {unit}')
print(f'lens k1 = {h[8]:.4f}' + ('' if fit_lens else ' (lens fit disabled)'))

# express the model in sub-stream (frame_size) pixels
sx = Wimg / fw
H = np.append(h[:8], 1).reshape(3, 3) @ np.diag([sx / S, sx / S, 1])
zone = [float(np.floor(lm_u.min())), float(np.ceil(lm_u.max()))]
cfg['calibration'] = {'_comment': f'fitted by tools/calibrate.py from {os.path.basename(pts_path)}',
                      'H': (H / np.linalg.norm(H)).tolist(), 'k1': float(h[8]), 'centre': [CX / sx, CY / sx],
                      'radius': RN / sx, 'zone_u': zone, 'road_width_m': round(W, 4)}
# furthest distance worth fitting: where one sub-stream pixel still covers <= 0.5 m of road (on the centre line)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CONFIG'] = os.path.abspath(cfg_path)
from speed import Road
_road = Road(cfg['calibration'])
max_u = zone[0]
for u in np.arange(zone[0], zone[1], 0.5):
    a_, b_ = np.array(_road.to_image(u, W / 2)), np.array(_road.to_image(u + 1, W / 2))
    if 1 / max(np.hypot(*(b_ - a_)), 1e-6) > 0.5:
        break
    max_u = float(u)
if P.get('road_polygon'):
    cfg['road_polygon'] = [[round(x / sx), round(y / (Himg / fh))] for x, y in P['road_polygon']]
else:
    print('warning: no road_polygon in points file; keeping the one in config.json')
cfg.setdefault('fit', {})['max_u'] = max_u
json.dump(cfg, open(cfg_path, 'w'), indent=2)
print(f'wrote calibration to {cfg_path}: measured zone u = {zone[0]:.0f}..{zone[1]:.0f} m; speeds are fitted out to '
      f'fit.max_u = {max_u:.1f} m, where a detection-stream pixel reaches 0.5 m of road.')

img_path = os.path.join(os.path.dirname(os.path.abspath(pts_path)), P.get('image', ''))
if P.get('image') and os.path.exists(img_path):
    import cv2
    road, img = _road, cv2.imread(img_path)
    th = max(1, round(sx))

    def Pt(u, v):
        x, y = road.to_image(u, v)
        return int(x * sx), int(y * sx)

    for v, col in ((0, (0, 255, 255)), (W / 2, (0, 165, 255)), (W, (0, 255, 255))):
        cv2.polylines(img, [np.array([Pt(u, v) for u in np.arange(zone[0] - 3, zone[1] + 3.01, 0.5)])], False, col, 2 * th)
    step = 5 if unit == 'm' else 10 * FT
    last = None
    for u in np.arange(np.ceil(zone[0] / step) * step, zone[1] + 0.01, step):
        cv2.line(img, Pt(u, -0.5), Pt(u, W + 0.5), (255, 0, 255), 2 * th)
        if last is None or abs(Pt(u, W)[1] - last) > 30 * th:  # skip labels where far-away lines bunch up
            cv2.putText(img, f'{u / scale:.0f} {unit}', Pt(u, W + 1.0), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, (255, 0, 255), th)
            last = Pt(u, W)[1]
    for x, y in lm:
        cv2.circle(img, (int(x), int(y)), 6 * th, (0, 0, 255), 2 * th)
    out = os.path.join(os.path.dirname(img_path), 'calib_check.jpg')
    cv2.imwrite(out, img)
    print(f'check image: {out} (grid lines should sit on the pavement edges, centre line and your landmarks)')
