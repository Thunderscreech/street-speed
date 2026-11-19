"""Speeding alerts: a rolling buffer of the camera's main (high-resolution) stream, stream-copied with no decoding,
clip cutting around a vehicle's pass, and posting the clip to Discord through a webhook or a bot."""
import glob, json, logging, os, subprocess, threading, time
import requests
from util import rtsp_url

log = logging.getLogger('alert')
BUF = '/buffer'


class Buffer:
    """ffmpeg copies the main stream into 2-second .ts segments named by their wall-clock start (seg_<epoch>.ts)."""

    def __init__(self, cam, keep_s=45):
        self.cam, self.keep_s = cam, keep_s
        os.makedirs(BUF, exist_ok=True)
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._prune, daemon=True).start()

    def _run(self):
        pw = os.environ.get('CAMERA_PASSWORD') or '\0'
        url = rtsp_url(self.cam, self.cam['clip_stream'])
        while True:
            p = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-rtsp_transport', 'tcp', '-i', url,
                                '-map', '0:v', '-c', 'copy', '-f', 'segment', '-segment_time', '2', '-reset_timestamps', '1',
                                '-strftime', '1', f'{BUF}/seg_%s.ts'], capture_output=True, text=True)
            log.warning('buffer ffmpeg exited (%s): %s', p.returncode, p.stderr.strip()[-300:].replace(pw, '***'))
            time.sleep(5)

    def _prune(self):
        while True:
            cutoff = time.time() - self.keep_s
            for f in glob.glob(f'{BUF}/seg_*.ts'):
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
            time.sleep(5)

    def clip(self, t0, t1, out):
        """Cut wall-clock window [t0, t1] out of the buffer into a 1080p H.264 MP4 under Discord's 10 MB upload limit."""
        segs = sorted((int(os.path.basename(f)[4:-3]), f) for f in glob.glob(f'{BUF}/seg_*.ts'))
        # a segment covers [its start, next segment's start); the newest one is still being written, so skip it
        chosen = [(s, f) for (s, f), (n, _) in zip(segs, segs[1:]) if n > t0 and s < t1]
        if not chosen:
            raise RuntimeError(f'no buffered video for {t0:.0f}-{t1:.0f} (have {segs[0][0] if segs else None}-{segs[-1][0] if segs else None})')
        lst = out + '.txt'
        with open(lst, 'w') as fh:
            fh.writelines(f"file '{f}'\n" for _, f in chosen)
        start = max(0.0, t0 - chosen[0][0])
        for height, crf in ((1080, 26), (720, 30)):
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst,
                            '-ss', f'{start:.2f}', '-t', f'{t1 - t0:.2f}', '-vf', f'scale=-2:{height}',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', '-an', out], check=True, capture_output=True)
            if os.path.getsize(out) < 9.5e6:
                break
        os.remove(lst)
        return out


class Discord:
    """Posts with a channel webhook (DISCORD_WEBHOOK_URL, simplest) or a bot (DISCORD_BOT_TOKEN + alert.channel)."""

    def __init__(self, channel=None):
        hook = os.environ.get('DISCORD_WEBHOOK_URL')
        if hook:
            self.url, self.h, self.check_url = hook, {}, hook
        else:
            self.url = f'https://discord.com/api/v10/channels/{channel}/messages'
            self.check_url = self.url.rsplit('/', 1)[0]
            self.h = {'Authorization': f"Bot {os.environ['DISCORD_BOT_TOKEN']}", 'User-Agent': 'DiscordBot (street-speed, 1.0)'}

    def post(self, text, path=None):
        files = {}
        if path:
            files['files[0]'] = (os.path.basename(path), open(path, 'rb'), 'video/mp4' if path.endswith('.mp4') else 'image/jpeg')
        r = requests.post(self.url, headers=self.h, data={'payload_json': json.dumps({'content': text})}, files=files, timeout=60)
        if r.status_code >= 300:
            raise RuntimeError(f'Discord {r.status_code}: {r.text[:200]}')


class Alerter:
    def __init__(self, cfg, cam, data_dir):
        self.cfg, self.dir = cfg, f'{data_dir}/clips'
        os.makedirs(self.dir, exist_ok=True)
        self.buffer = Buffer(cam)
        self.discord = Discord(cfg.get('channel'))
        r = requests.get(self.discord.check_url, headers=self.discord.h, timeout=10)
        if r.ok:
            log.info('Discord alerts go to %s (>= %s)', r.json().get('name'), cfg['speed'])
        else:
            log.warning('cannot reach the Discord channel/webhook (%s %s); alerts will fail until that is fixed',
                        r.status_code, r.text[:120])

    def maybe_alert(self, m, wall_start, wall_end, snapshot=None):
        c = self.cfg
        if m['speed'] < c['speed']:
            return
        # fast cars cross the measured stretch in under a second, so judge quality by distance covered and fit error,
        # not by point count (at highway speeds a car is in the measured stretch for only ~10 frames)
        if m['points'] < c['min_points'] or m['span_m'] < c.get('min_span_m', 0) or m['rms'] > c['max_rms_m']:
            log.info('%.0f %s but the track is too rough for an alert (n=%d span=%.1fm rms=%.2f)',
                     m['speed'], m['unit'], m['points'], m['span_m'], m['rms'])
            return
        threading.Thread(target=self._send, args=(m, wall_start, wall_end, snapshot), daemon=True).start()

    def _send(self, m, a, b, snapshot):
        import datetime as dt
        from zoneinfo import ZoneInfo
        # detection times come from the sub-stream, which usually reaches us later than the main stream the buffer
        # records (on a Reolink E1: sub ~2.8 s behind the camera clock, main ~1 s), so shift the window back by that
        lag = self.cfg.get('stream_lag_s', 0)
        t0, t1 = a - lag - self.cfg['pre_s'], b - lag + self.cfg['post_s']
        time.sleep(max(0, t1 + 5 - time.time()))  # let the segment holding the end close
        when = dt.datetime.fromtimestamp(b, ZoneInfo(os.environ.get('TZ', 'America/Los_Angeles')))
        heading = 'away from the camera' if m['direction'] == 'away' else 'toward the camera'
        text = f"🚨 **{m['speed']:.0f} {m['unit']}** {m['cls']} heading {heading} · {when:%a %b %-d, %-I:%M:%S %p}"
        out = f"{self.dir}/{when:%Y%m%d_%H%M%S}_{m['speed']:.0f}{m['unit'].replace('/', '')}.mp4"
        try:
            self.buffer.clip(t0, t1, out)
            self.discord.post(text, out)
            log.info('alert sent: %s (%.1f MB)', out, os.path.getsize(out) / 1e6)
        except Exception as e:
            log.warning('alert with clip failed (%s); sending snapshot instead', e)
            try:
                self.discord.post(text + ' (no video)', snapshot)
            except Exception as e2:
                log.error('alert failed: %s', e2)
