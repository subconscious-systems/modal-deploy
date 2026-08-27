# Thin serving layer on top of the heavy timrun base image.
#
# Stage 1: build the base from timrun-sys/docker/Dockerfile and push it to
#          Docker Hub as the base tag.
# Stage 2: build THIS Dockerfile (FROM that base) and push as the serve tag.
#          This is not used when deploy.py pulls the prebuilt Hub image.
#
# Build:
#   docker build \
#     --build-arg BASE_IMAGE=subconsciouslabs/sglang-baseten:sm_100-v0.12 \
#     -t subconsciouslabs/sglang-baseten:sm_100-v0.12-serve .

ARG BASE_IMAGE=subconsciouslabs/sglang-baseten:sm_100-v0.12
FROM ${BASE_IMAGE}

# Modal's runtime expects `python` and `pip` on PATH. The timrun base image
# installs python3.12 via update-alternatives but may not alias `python`/`pip`.
RUN ln -sf "$(which python3)" /usr/local/bin/python 2>/dev/null || true \
 && ln -sf "$(which pip3)"    /usr/local/bin/pip    2>/dev/null || true

# Serving defaults (overridable by Modal at runtime). MODEL_PATH points into
# the Modal Volume mounted at /models/tim_model.
ENV MODEL_PATH=/models/tim_model/SubconsciousDev/TIM-8b-long-grpo \
    SGLANG_PORT=30000 \
    HOST=0.0.0.0

# No ENTRYPOINT/CMD: Modal's @modal.web_server supplies the launch command
# (python3 -m sglang.launch_server ...). If you want to run this image
# standalone under docker, override the command, e.g.:
#   docker run --gpus all -p 30000:30000 <image> \
#     python3 -m sglang.launch_server --model-path $MODEL_PATH --host 0.0.0.0 --port 30000
