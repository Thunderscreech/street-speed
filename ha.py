"""Home Assistant access: REST for live sensor states, websocket for importing hourly/daily statistics."""
import json, logging, threading
import requests, websocket

log = logging.getLogger('ha')


class HA:
    def __init__(self, url, token):
        self.url = url.rstrip('/')
        self.token = token
        self.s = requests.Session()
        self.s.headers['Authorization'] = f'Bearer {token}'
        self._ws_lock = threading.Lock()

    def get_state(self, entity_id):
        r = self.s.get(f'{self.url}/api/states/{entity_id}', timeout=5)
        r.raise_for_status()
        return r.json()

    def set_state(self, entity_id, state, attributes):
        r = self.s.post(f'{self.url}/api/states/{entity_id}', json={'state': state, 'attributes': attributes}, timeout=5)
        r.raise_for_status()

    def _ws(self, messages):
        """Open a websocket, authenticate, send each message and return the results in order."""
        ws_url = self.url.replace('http', 'ws', 1) + '/api/websocket'
        with self._ws_lock:
            ws = websocket.create_connection(ws_url, timeout=15)
            try:
                json.loads(ws.recv())  # auth_required
                ws.send(json.dumps({'type': 'auth', 'access_token': self.token}))
                auth = json.loads(ws.recv())
                if auth.get('type') != 'auth_ok':
                    raise RuntimeError(f'HA websocket auth failed: {auth}')
                out = []
                for i, m in enumerate(messages, 1):
                    ws.send(json.dumps({'id': i, **m}))
                    while True:
                        r = json.loads(ws.recv())
                        if r.get('id') == i and r.get('type') == 'result':
                            break
                    if not r.get('success'):
                        raise RuntimeError(f'HA websocket {m["type"]} failed: {r.get("error")}')
                    out.append(r.get('result'))
                return out
            finally:
                ws.close()

    def import_mean_stats(self, statistic_id, name, unit, rows):
        """rows: [{'start': iso hour start with tz, 'mean':, 'min':, 'max':}]. Rows with the same start are replaced."""
        meta = {'source': statistic_id.split(':')[0], 'statistic_id': statistic_id, 'name': name,
                'unit_of_measurement': unit, 'unit_class': 'speed', 'has_sum': False, 'mean_type': 1}
        self._ws([{'type': 'recorder/import_statistics', 'metadata': meta, 'stats': rows}])

    def ws_call(self, msg):
        return self._ws([msg])[0]
