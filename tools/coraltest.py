"""Quick check: run the vehicle detector on one image and time it on the Edge TPU and on the CPU."""
import sys, time
import cv2
from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file

img = cv2.imread(sys.argv[1])
labels = read_label_file('/app/models/coco_labels.txt')
print('edge TPUs:', list_edge_tpus())
for name, path, dev in [('tpu', '/app/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 'usb'),
                        ('cpu', '/app/models/ssdlite_mobiledet_coco_qat_postprocess.tflite', None)]:
    if dev:
        it = make_interpreter(path, device=dev)
    else:
        import tflite_runtime.interpreter as tfl
        it = tfl.Interpreter(path, num_threads=1)
    it.allocate_tensors()
    w, h = common.input_size(it)
    rgb = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
    times = []
    for _ in range(20):
        t = time.perf_counter(); common.set_input(it, rgb); it.invoke(); times.append(time.perf_counter() - t)
    objs = detect.get_objects(it, 0.4)
    print(name, f'median {sorted(times)[10]*1000:.1f} ms', [(labels.get(o.id), round(o.score, 2), o.bbox) for o in objs])
