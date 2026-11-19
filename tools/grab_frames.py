"""Save one frame from each camera stream, for calibration and for checking the setup.

usage (inside the container): python3 tools/grab_frames.py /data/calib
Writes frame_main.jpg (full resolution; pick calibration points on this one) and frame_sub.jpg (the detection
stream; its size is what config.json frame_size must be)."""
import os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed import CFG
from util import rtsp_url

out = sys.argv[1] if len(sys.argv) > 1 else '/data/calib'
os.makedirs(out, exist_ok=True)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cam = CFG['camera']
for name, path in (('main', cam['clip_stream']), ('sub', cam['stream'])):
    cap = cv2.VideoCapture(rtsp_url(cam, path), cv2.CAP_FFMPEG)
    ok, frame = False, None
    for _ in range(15):  # skip the first frames, which can be grey until a keyframe arrives
        ok, f = cap.read()
        frame = f if ok else frame
    if frame is None:
        print(f'{name}: could not read stream "{path}"'); continue
    cv2.imwrite(f'{out}/frame_{name}.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'{name}: {frame.shape[1]}x{frame.shape[0]} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps -> {out}/frame_{name}.jpg')
