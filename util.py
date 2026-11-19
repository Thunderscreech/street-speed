"""Small shared helpers."""
import os
from urllib.parse import quote


def rtsp_url(cam, path):
    """rtsp://user:password@host:port/path, with the password from the CAMERA_PASSWORD environment variable.
    `path` is the stream path after the port, which differs by camera brand (see README)."""
    user = quote(cam.get('user', ''), safe='')
    pw = quote(os.environ.get('CAMERA_PASSWORD', ''), safe='')
    auth = f'{user}:{pw}@' if user else ''
    return f"rtsp://{auth}{cam['host']}:{cam.get('port', 554)}/{path.lstrip('/')}"
