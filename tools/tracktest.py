"""Live test without calibration: run the motion gate, detector and tracker on the stream for N seconds and
print every finished track (frames, duration, pixel path). Saves one contact sheet per track to /out."""
import sys, time, cv2, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import speed

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 180
det = speed.Detector(speed.CFG['detector'])
cap = speed.open_stream()
m = speed.CFG['motion']
mask = speed.road_mask()
tracks, prev, active_until = [], None, 0
t_end, det_t, nframes, ndet = time.time() + secs, 0.0, 0, 0
t_start = time.time()
while time.time() < t_end:
    ok, frame = cap.read()
    if not ok:
        print('read failed'); break
    nframes += 1
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    g = speed.small_gray(frame)
    if prev is not None and speed.road_motion(g, prev, mask):
        active_until = t + m['hold_s']
    prev = g
    if t <= active_until or tracks:
        a = time.perf_counter(); dets = [d for d in det(frame) if speed.on_road(d)]; det_t += time.perf_counter() - a; ndet += 1
        matches, new = speed.associate(tracks, dets, t)
        for ti, di in matches:
            tracks[ti].add(dets[di], t)
            if len(tracks[ti].obs) % 4 == 1:
                tracks[ti].__dict__.setdefault('frames', []).append(frame.copy())
        for d in new:
            tr = speed.Track(d, t, g); tr.frames = [frame.copy()]; tracks.append(tr)
        for tr in [tr for tr in tracks if t - tr.last_t > 0.8]:
            tracks.remove(tr)
            o = tr.obs
            if len(o) < 4:
                continue
            path = ' '.join(f'({x:.0f},{y:.0f})' for _, x, y, *_ in o[::max(1, len(o) // 6)])
            print(f'track {tr.id}: {len(o)} pts over {o[-1][0]-o[0][0]:.2f}s cls={o[0][4]} path {path}', flush=True)
            img = tr.frames[len(tr.frames) // 2].copy()
            for _, x, y, *_ in o:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.imwrite(f'/out/track_{tr.id}.jpg', img)
el = time.time() - t_start
print(f'{nframes/el:.1f} fps, detector ran on {ndet} frames, busy {100*det_t/el:.0f}%')
