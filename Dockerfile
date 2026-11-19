# Street speed service. pycoral 2.0 / tflite-runtime 2.5 wheels exist only for Python <= 3.9, so the base is 3.9.
FROM python:3.9-slim-bookworm
ENV PYTHONUNBUFFERED=1
ARG LIBEDGETPU_DEB=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.bookworm_amd64.deb
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Edge TPU runtime. Google's Coral apt repository is gone; this uses the community build of libedgetpu by default.
# To use your own copy instead (e.g. from a host with libedgetpu1-std installed), put libedgetpu.so.1 in vendor/.
COPY vendor/ /tmp/vendor/
RUN if [ -f /tmp/vendor/libedgetpu.so.1 ]; then \
        cp /tmp/vendor/libedgetpu.so.1 /usr/lib/x86_64-linux-gnu/libedgetpu.so.1; \
    else \
        curl -fsSL -o /tmp/edgetpu.deb "$LIBEDGETPU_DEB" && dpkg -x /tmp/edgetpu.deb /tmp/edgetpu \
        && cp -P /tmp/edgetpu/usr/lib/x86_64-linux-gnu/libedgetpu.so* /usr/lib/x86_64-linux-gnu/ && rm -rf /tmp/edgetpu*; \
    fi && ldconfig && rm -rf /tmp/vendor
RUN pip install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ \
      "numpy<1.24" pycoral==2.0.0 tflite-runtime==2.5.0.post1 \
      "opencv-python-headless==4.8.1.78" requests websocket-client
WORKDIR /app
COPY models /app/models
COPY *.py /app/
COPY tools /app/tools
CMD ["python3", "/app/speed.py"]
