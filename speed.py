"""Street speed service.

Reads a fixed street camera's RTSP sub-stream, detects vehicles on a Coral Edge TPU while there is motion on the
road, tracks each vehicle through every frame it is visible in, maps its ground point onto the road plane with a
homography calibrated from measured landmarks, and fits distance-along-road against frame time to get its speed.
Results go to Home Assistant as live sensors (REST) and as hourly/daily mean statistics (recorder import).
"""
import datetime as dt, json, logging, os, sqlite3, threading, time
from zoneinfo import ZoneInfo
import cv2, numpy as np
from ha import HA
from util import rtsp_url

log = logging.getLogger('speed')
CFG = json.load(open(os.environ.get('CONFIG', '/app/config.json')))
UNIT = CFG.get('speed_unit', 'mph')  # 'mph' or 'km/h'
MPS_TO_UNIT = {'mph': 2.236936, 'km/h': 3.6}[UNIT]
DATA = os.environ.get('DATA_DIR', '/data')
TZ = ZoneInfo(os.environ.get('TZ', 'America/Los_Angeles'))
PROGRESS = {'t': time.time()}  # last time the frame loop made progress; see watchdog()


def watchdog(limit_s=60):
    """The USB Coral can hang inside invoke() (and an RTSP read can block) without raising. If the frame loop stops
    for limit_s, exit so Docker's restart policy brings the service back with a freshly opened TPU and stream."""
    while True:
        time.sleep(10)
        stalled = time.time() - PROGRESS['t']
        if stalled > limit_s:
            log.error('frame loop stalled for %.0f s (Coral or stream hang); exiting for a restart', stalled)
            logging.shutdown()
            os._exit(3)


# ---------------------------------------------------------------- calibration
class Road:
    """Image (sub-stream px) -> road plane (u metres along, v metres across), via a one-term radial lens
    correction and a homography, both fitted by calibfit.py."""

    def __init__(self, cal):
        self.H = np.array(cal['H'], dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.k1, self.c, self.rn = cal['k1'], np.array(cal['centre'], float), cal['radius']
        self.zone = cal['zone_u']

    def undistort(self, pts):
        d = (np.asarray(pts, float) - self.c) / self.rn
        return self.c + d * (1 + self.k1 * (d ** 2).sum(-1, keepdims=True)) * self.rn

    def distort(self, pts):
        """Inverse of undistort (fixed-point iteration), for drawing road lines onto the image."""
        u = (np.asarray(pts, float) - self.c) / self.rn
        d = u.copy()
        for _ in range(30):
            d = u / (1 + self.k1 * (d ** 2).sum(-1, keepdims=True))
        return self.c + d * self.rn

    def to_road(self, x, y):
        p = self.H @ np.append(self.undistort([x, y]), 1)
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_image(self, u, v):
        p = self.Hinv @ [u, v, 1]
        return self.distort(p[:2] / p[2])


# ---------------------------------------------------------------- detection
class Detector:
    def __init__(self, cfg):
        from pycoral.adapters import common, detect
        from pycoral.utils.dataset import read_label_file
        self.common, self.detect = common, detect
        device = cfg.get('device', 'usb')  # 'usb' or 'pci' for a Coral Edge TPU, 'cpu' to run without one
        if device == 'cpu':
            import tflite_runtime.interpreter as tfl
            self.it = tfl.Interpreter(cfg['cpu_model'], num_threads=cfg.get('cpu_threads', 2))
        else:
            from pycoral.utils.edgetpu import make_interpreter
            self.it = make_interpreter(cfg['model'], device=device)
        log.info('detector on %s', device)
        self.it.allocate_tensors()
        self.size = common.input_size(self.it)
        labels = read_label_file(cfg['labels'])
        self.keep = {i: n for i, n in labels.items() if n in cfg['classes']}
        self.min_score = cfg['min_score']

    def __call__(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2RGB)
        self.common.set_input(self.it, rgb)
        self.it.invoke()
        sx, sy = w / self.size[0], h / self.size[1]
        out = []
        for o in self.detect.get_objects(self.it, self.min_score):
            if o.id in self.keep:
                b = o.bbox
                out.append({'cls': self.keep[o.id], 'score': float(o.score),
                            'box': (b.xmin * sx, b.ymin * sy, b.xmax * sx, b.ymax * sy)})
        return nms(out)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def nms(dets, thr=0.5):
    """Class-agnostic: the detector sometimes calls one vehicle both car and truck."""
    dets = sorted(dets, key=lambda d: -d['score']); keep = []
    for d in dets:
        if all(iou(d['box'], k['box']) < thr for k in keep):
            keep.append(d)
    return keep


# ---------------------------------------------------------------- tracking
class Track:
    _next = 1

    def __init__(self, det, t, gray):
        self.id = Track._next; Track._next += 1
        self.obs = []  # (t, ground_x, ground_y, box, cls, score)
        self.first_gray = gray
        self.best = None  # (area, frame, box) for the event snapshot
        self.add(det, t)

    def add(self, det, t):
        x0, y0, x1, y1 = det['box']
        # (t, ground_x, ground_y, box, cls, score, u, v, box_cut_by_frame_edge)
        self.obs.append((t, (x0 + x1) / 2, y1, det['box'], det['cls'], det['score'], det['u'], det['v'], det['edge']))
        self.last_t = t

    def predict(self, t):
        """Predicted road position (u, v) at time t from the recent velocity along the road."""
        o = self.obs
        if len(o) < 2:
            return o[-1][6], o[-1][7], None
        k = max(0, len(o) - 4)  # velocity over the last few points, so one noisy box doesn't fling the prediction
        vel = (o[-1][6] - o[k][6]) / max(o[-1][0] - o[k][0], 1e-3)
        return o[-1][6] + vel * (t - o[-1][0]), o[-1][7], vel


def associate(tracks, dets, t):
    """Match detections to tracks by distance on the road plane (metres), not pixels: far away a whole car is
    ~10 px, so a pixel gate lets a track jump between cars. The gate widens with distance (coarser resolution)."""
    pairs = []
    for ti, tr in enumerate(tracks):
        pu, pv, vel = tr.predict(t)
        for di, d in enumerate(dets):
            du, dv = abs(d['u'] - pu), abs(d['v'] - pv)
            gate_u = (2.0 if vel is not None else 6.0) + 0.1 * max(0.0, d['u'])
            if du < gate_u and dv < 2.0:
                pairs.append((du / gate_u + dv / 2.0, ti, di))
    used_t, used_d, matches = set(), set(), []
    for dist, ti, di in sorted(pairs):
        if ti not in used_t and di not in used_d:
            used_t.add(ti); used_d.add(di); matches.append((ti, di))
    return matches, [d for i, d in enumerate(dets) if i not in used_d]


# ---------------------------------------------------------------- road area
ROAD_POLY = np.array(CFG['road_polygon'], dtype=np.float32)


def on_road(det):
    x0, y0, x1, y1 = det['box']
    return cv2.pointPolygonTest(ROAD_POLY, ((x0 + x1) / 2, y1), False) >= 0


def road_mask():
    sw, sh = CFG['motion']['scale']; fw, fh = CFG['frame_size']
    m = np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(m, [np.round(ROAD_POLY * [sw / fw, sh / fh]).astype(np.int32)], 1)
    return m.astype(bool)


def road_motion(g, prev, mask):
    m = CFG['motion']
    return int(((cv2.absdiff(g, prev) > m['pixel_threshold']) & mask).sum()) >= m['min_pixels']


# ---------------------------------------------------------------- camera-moved guard
def small_gray(frame):
    g = cv2.cvtColor(cv2.resize(frame, tuple(CFG['motion']['scale']), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


_STATIC = None


def shift_px(ref, g):
    """Image shift between two small grays, in full-frame pixels, measured on the scenery only: the road and the
    space above it (where passing cars are) are blanked identically in both images, so traffic can't look like a
    camera move."""
    global _STATIC
    if _STATIC is None:
        m = road_mask().astype(np.uint8)
        m = cv2.dilate(m, np.ones((15, 5), np.uint8), anchor=(2, 14))  # extend upward over car bodies
        _STATIC = m.astype(bool)
    fill = float(ref[~_STATIC].mean())
    a, b = ref.copy(), g.copy()
    a[_STATIC] = fill; b[_STATIC] = fill
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(np.hypot(dx, dy)) * CFG['frame_size'][0] / CFG['motion']['scale'][0]


# ---------------------------------------------------------------- storage + HA publishing
class Store:
    def __init__(self, ha):
        self.ha = ha
        os.makedirs(f'{DATA}/events', exist_ok=True)
        self.db = sqlite3.connect(f'{DATA}/speeds.db', check_same_thread=False)
        self.db.execute('create table if not exists speeds (ts real, speed real, direction text, cls text, '
                        'points int, rms real, span_m real, snapshot text)')
        self.db.execute('create table if not exists rejects (ts real, reason text, detail text)')
        self.db.commit()
        self.lock = threading.Lock()
        self.last = None
        row = self.db.execute('select ts, speed, direction, cls from speeds order by ts desc limit 1').fetchone()
        if row:
            self.last = dict(zip(('ts', 'speed', 'direction', 'cls'), row))

    def reject(self, reason, detail=''):
        self._last_reject = f'{reason} {detail}'
        with self.lock:
            self.db.execute('insert into rejects values (?,?,?)', (time.time(), reason, detail)); self.db.commit()
        log.info('rejected: %s %s', reason, detail)
        return self._last_reject

    def add(self, m):
        with self.lock:
            self.db.execute('insert into speeds values (?,?,?,?,?,?,?,?)',
                            (m['ts'], m['speed'], m['direction'], m['cls'], m['points'], m['rms'], m['span_m'], m['snapshot']))
            self.db.commit()
        self.last = m
        self.publish(import_stats=True, ts=m['ts'])

    def _agg(self, t0, t1):
        return self.db.execute('select avg(speed), min(speed), max(speed), count(*) from speeds where ts>=? and ts<?',
                               (t0, t1)).fetchone()

    def _max(self, t0):
        return self.db.execute('select ts, speed, direction, cls from speeds where ts>=? order by speed desc limit 1',
                               (t0,)).fetchone()

    def publish(self, import_stats=False, ts=None):
        if os.environ.get('DRY_RUN'):
            return
        now = dt.datetime.now(TZ)
        hour0 = now.replace(minute=0, second=0, microsecond=0)
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.lock:
            h = self._agg(hour0.timestamp(), hour0.timestamp() + 3600)
            d = self._agg(day0.timestamp(), day0.timestamp() + 86400)
            week0 = day0 - dt.timedelta(days=day0.weekday())  # weeks start on Monday
            month0 = day0.replace(day=1)
            maxima = [(k, label, self._max(t.timestamp())) for k, label, t in
                      (('today', 'today', day0), ('this_week', 'this week', week0), ('this_month', 'this month', month0))]
        base = {'unit_of_measurement': UNIT, 'device_class': 'speed', 'icon': 'mdi:speedometer'}
        r1 = lambda v: round(v, 1) if v is not None else 'unknown'
        try:
            if self.last:
                when = dt.datetime.fromtimestamp(self.last['ts'], TZ)
                self.ha.set_state('sensor.street_speed_last_vehicle', r1(self.last['speed']), {
                    **base, 'friendly_name': 'Last vehicle speed', 'direction': self.last['direction'],
                    'vehicle': self.last['cls'], 'measured_at': when.isoformat(timespec='seconds')})
            self.ha.set_state('sensor.street_speed_avg_this_hour', r1(h[0]),
                              {**base, 'friendly_name': 'Average speed this hour', 'vehicles': h[3]})
            self.ha.set_state('sensor.street_speed_avg_today', r1(d[0]), {
                **base, 'friendly_name': 'Average speed today', 'vehicles': d[3],
                'min': r1(d[1]), 'max': r1(d[2])})
            self.ha.set_state('sensor.street_speed_measured_today', d[3], {
                'friendly_name': 'Vehicles speed-measured today', 'icon': 'mdi:car-speed-limiter', 'unit_of_measurement': 'vehicles'})
            for key, label, row in maxima:
                attrs = {**base, 'friendly_name': f'Top speed {label}', 'icon': 'mdi:speedometer-medium'}
                if row:
                    attrs.update(measured_at=dt.datetime.fromtimestamp(row[0], TZ).isoformat(timespec='seconds'),
                                 direction=row[2], vehicle=row[3])
                self.ha.set_state(f'sensor.street_speed_max_{key}', r1(row[1]) if row else 'unknown', attrs)
            if import_stats:
                self.import_stats(dt.datetime.fromtimestamp(ts, TZ))
        except Exception as e:
            log.warning('HA publish failed: %s', e)

    def import_stats(self, when):
        """Hourly statistic = per-vehicle mean/min/max for that clock hour; daily statistic = one row at local
        midnight holding the per-vehicle mean for the whole day, so a statistics-graph with period: day shows it."""
        hour0 = when.replace(minute=0, second=0, microsecond=0)
        day0 = when.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.lock:
            h = self._agg(hour0.timestamp(), hour0.timestamp() + 3600)
            d = self._agg(day0.timestamp(), day0.timestamp() + 86400)
        if h[3]:
            self.ha.import_mean_stats('street_speed:avg_speed_hourly', 'Street average speed (hourly)', UNIT,
                                      [{'start': hour0.isoformat(), 'mean': h[0], 'min': h[1], 'max': h[2]}])
        if d[3]:
            self.ha.import_mean_stats('street_speed:avg_speed_daily', 'Street average speed (daily)', UNIT,
                                      [{'start': day0.isoformat(), 'mean': d[0], 'min': d[1], 'max': d[2]}])

    def prune(self):
        cutoff = time.time() - CFG['keep_days'] * 86400
        for p in [f'{DATA}/events/{f}' for f in os.listdir(f'{DATA}/events')] + \
                 [f'{DATA}/clips/{f}' for f in (os.listdir(f'{DATA}/clips') if os.path.isdir(f'{DATA}/clips') else [])]:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)


# ---------------------------------------------------------------- measurement
def dump_track(track, road, outcome, frame=None):
    """Debug record of a finished track (points, road coords, outcome) and the frame with its path drawn."""
    if not CFG.get('debug_tracks'):
        return
    d = f'{DATA}/tracks'; os.makedirs(d, exist_ok=True)
    name = dt.datetime.fromtimestamp(track.wall_end, TZ).strftime('%Y%m%d_%H%M%S') + f'_{track.id}'
    rows = [{'t': o[0], 'x': o[1], 'y': o[2], 'box': list(o[3]), 'cls': o[4], 'score': o[5],
             'u': float(road.to_road(o[1], o[2])[0]), 'v': float(road.to_road(o[1], o[2])[1])} for o in track.obs]
    json.dump({'id': track.id, 'outcome': outcome, 'obs': rows}, open(f'{d}/{name}.json', 'w'))
    img = (track.best[1] if track.best is not None else frame)
    if img is not None:
        img = img.copy()
        for i, o in enumerate(track.obs):
            x0, y0, x1, y1 = map(int, o[3])
            c = (0, int(255 * i / max(1, len(track.obs) - 1)), 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
            cv2.circle(img, (int(o[1]), int(o[2])), 2, c, -1)
        cv2.putText(img, outcome[:60], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(f'{d}/{name}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def measure(track, road, store, gray_now, ref, pan_ok, alerter=None, frame=None):
    if len(track.obs) < CFG['fit']['min_points']:
        return  # too short to be worth logging (mostly far-away or parked blips)
    res = _measure(track, road, store, gray_now, ref, pan_ok, alerter)
    try:
        dump_track(track, road, res or 'rejected', frame)
    except Exception as e:
        log.warning('track dump failed: %s', e)


def _measure(track, road, store, gray_now, ref, pan_ok, alerter=None):
    obs = track.obs
    f = CFG['fit']
    t = np.array([o[0] for o in obs])
    u = np.array([o[6] for o in obs])
    edge = np.array([o[8] for o in obs])
    z0, z1 = road.zone
    # beyond fit max_u a pixel is ~1 m of road, too coarse to fit; boxes cut by the frame edge have a wrong ground point
    inside = (u >= z0 - 3) & (u <= min(z1, f.get('max_u', z1))) & ~edge
    t, u = t[inside], u[inside]
    tag = f'track {track.id} n={len(obs)}'
    if len(t) < f['min_points']:
        return store.reject('outside_zone', tag)
    if ref is not None:
        s = max(shift_px(ref, track.first_gray), shift_px(ref, gray_now))
        if s > CFG['camera_moved_px']:
            return store.reject('camera_moved', f'{tag} shift={s:.1f}px')
    if not pan_ok():
        return store.reject('camera_not_home', tag)
    dur, span = t[-1] - t[0], u.max() - u.min()
    if dur < f['min_duration_s'] or span < f['min_span_m']:
        return store.reject('too_short', f'{tag} dur={dur:.2f}s span={span:.1f}m')
    # robust straight-line fit: drop single bad boxes (occlusion, a box merging two cars) then refit
    keep = np.ones(len(t), bool)
    for _ in range(3):
        A = np.vstack([t[keep] - t[0], np.ones(keep.sum())]).T
        (slope, icpt), *_ = np.linalg.lstsq(A, u[keep], rcond=None)
        r = u - (slope * (t - t[0]) + icpt)
        mad = 1.4826 * np.median(np.abs(r[keep]))
        new = np.abs(r) <= max(3 * mad, f.get('outlier_floor_m', 0.75))
        if (new == keep).all():
            break
        keep = new
    n_in = int(keep.sum())
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    speed = abs(slope) * MPS_TO_UNIT
    if n_in < max(f['min_points'], f.get('min_inlier_frac', 0.6) * len(t)) or rms > f['max_rms_m'] \
            or not (f['min_speed'] <= speed <= f['max_speed']):
        return store.reject('bad_fit', f'{tag} speed={speed:.1f} rms={rms:.2f}m inliers={n_in}/{len(t)}')
    t, u = t[keep], u[keep]
    cls = max(set(o[4] for o in obs), key=lambda c: sum(1 for o in obs if o[4] == c))
    direction = 'away' if slope > 0 else 'toward'
    ts = track.wall_end
    snap = ''
    if track.best is not None:
        _, frame, box = track.best
        img = frame.copy()
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        for o in obs:
            cv2.circle(img, (int(o[1]), int(o[2])), 2, (0, 0, 255), -1)
        cv2.putText(img, f'{speed:.0f} {UNIT} {direction}', (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        snap = dt.datetime.fromtimestamp(ts, TZ).strftime('%Y%m%d_%H%M%S') + f'_{track.id}.jpg'
        cv2.imwrite(f'{DATA}/events/{snap}', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info('%s: %.1f %s %s (%s) n=%d dur=%.2fs span=%.1fm rms=%.2fm seen %s-%s', tag, speed, UNIT, direction, cls, len(t),
             dur, span, rms, *(dt.datetime.fromtimestamp(x, TZ).strftime('%H:%M:%S.%f')[:-4] for x in (track.wall_start, track.wall_end)))
    m = {'ts': ts, 'speed': speed, 'unit': UNIT, 'direction': direction, 'cls': cls, 'points': int(len(t)),
         'rms': rms, 'span_m': float(span), 'snapshot': snap}
    store.add(m)
    if alerter:
        alerter.maybe_alert(m, track.wall_start, track.wall_end, f'{DATA}/events/{snap}' if snap else None)
    return f'ok {speed:.1f} {UNIT} rms={rms:.2f}m'


# ---------------------------------------------------------------- main loop
def open_stream():
    c = CFG['camera']
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(rtsp_url(c, c['stream']), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open RTSP stream {c['host']}/{c['stream']}")
    return cap


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    token = os.environ.get('HA_TOKEN') or open(os.environ.get('HA_TOKEN_FILE', '/run/secrets/ha_token')).read().strip()
    ha = HA(CFG['ha_url'], token)
    if not CFG.get('calibration', {}).get('H') or len(CFG.get('road_polygon', [])) < 3:
        raise SystemExit('config.json has no calibration yet: run tools/grab_frames.py, mark points with '
                         'tools/picker.html and run tools/calibrate.py (see README)')
    store = Store(ha)
    road = Road(CFG['calibration'])
    det = Detector(CFG['detector'])
    mcfg = CFG['motion']
    alerter = None
    if CFG.get('alert', {}).get('enabled'):
        from alert import Alerter
        alerter = Alerter(CFG['alert'], CFG['camera'], DATA)

    pan_cache = {'t': 0, 'ok': True}

    def pan_ok():
        """For PTZ cameras: only trust readings while an HA sensor shows the camera at its calibrated position."""
        if not CFG['camera'].get('pan_entity'):
            return True
        if time.time() - pan_cache['t'] > 10:
            try:
                pan_cache['ok'] = ha.get_state(CFG['camera']['pan_entity'])['state'] == str(CFG['camera']['home_pan'])
            except Exception as e:
                log.warning('pan check failed: %s', e)
            pan_cache['t'] = time.time()
        return pan_cache['ok']

    def heartbeat():
        while True:
            store.publish(); store.prune(); time.sleep(60)
    threading.Thread(target=heartbeat, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    while True:
        try:
            PROGRESS['t'] = time.time()
            cap = open_stream()
            log.info('stream open: %dx%d @ %.1f fps', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
            loop(cap, road, det, store, mcfg, pan_ok, alerter)
        except Exception as e:
            log.exception('stream loop error: %s', e)
        time.sleep(5)


def loop(cap, road, det, store, mcfg, pan_ok, alerter=None):
    tracks, prev, ref, ref_t = [], None, None, 0
    active_until, pts0, wall0 = 0.0, None, None
    frames, t_stat, det_time = 0, time.time(), 0.0
    W, H = CFG['frame_size']
    mask = road_mask()
    while True:
        ok, frame = cap.read()
        PROGRESS['t'] = time.time()
        if not ok:
            raise RuntimeError('frame read failed')
        if frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        now = time.time()
        if pts0 is None or pts < pts0 - 1:
            pts0, wall0 = pts, now
        # wall clock of the stream timeline: the frame that arrived with the least delay sets it, so a backlog at
        # startup (first frames read seconds late) doesn't make every later timestamp, and clip window, late
        wall0 = min(wall0, now - (pts - pts0))
        t = pts  # frame time from the stream's own timestamps: what the speed depends on
        frames += 1
        g = small_gray(frame)

        # motion gate
        moving = prev is not None and road_motion(g, prev, mask)
        prev = g
        if moving:
            active_until = t + mcfg['hold_s']

        # reference frame for the camera-moved check: refreshed every 10 min while the road is quiet and the camera is home
        if not tracks and t > active_until and (ref is None or time.time() - ref_t > 600):
            if pan_ok():
                if ref is None or shift_px(ref, g) < 0.5 * CFG['camera_moved_px'] or time.time() - ref_t > 3600:
                    ref, ref_t = g.copy(), time.time()

        if t <= active_until or tracks:
            t1 = time.perf_counter()
            dets = [d for d in det(frame) if on_road(d)]
            for d in dets:
                b = d['box']
                d['u'], d['v'] = map(float, road.to_road((b[0] + b[2]) / 2, b[3]))
                d['edge'] = b[0] <= 1 or b[2] >= W - 2 or b[3] >= H - 2
            det_time += time.perf_counter() - t1
            matches, new = associate(tracks, dets, t)
            for ti, di in matches:
                tracks[ti].add(dets[di], t)
                b = dets[di]['box']; area = (b[2] - b[0]) * (b[3] - b[1])
                u, _ = road.to_road((b[0] + b[2]) / 2, b[3])
                z0, z1 = road.zone
                if z0 <= u <= z1 and (tracks[ti].best is None or area > tracks[ti].best[0]):
                    tracks[ti].best = (area, frame, b)
            for d in new:
                tracks.append(Track(d, t, g))
            for tr in [tr for tr in tracks if t - tr.last_t > 0.8]:
                tr.wall_start, tr.wall_end = wall0 + (tr.obs[0][0] - pts0), wall0 + (tr.last_t - pts0)
                tracks.remove(tr)
                try:
                    measure(tr, road, store, g, ref, pan_ok, alerter, frame)
                except Exception as e:
                    log.exception('measure failed: %s', e)

        if time.time() - t_stat > 300:
            log.info('stats: %.1f fps decoded, detector busy %.0f%%, %d tracks, frame delay %.1f s', frames / (time.time() - t_stat),
                     100 * det_time / (time.time() - t_stat), len(tracks), now - (wall0 + pts - pts0))
            frames, det_time, t_stat = 0, 0.0, time.time()


if __name__ == '__main__':
    run()
