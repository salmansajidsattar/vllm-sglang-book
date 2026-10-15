# You must give the exact vLLM image version you tested in Steps 3 to 6:
#   docker build --build-arg VLLM_VERSION=<the version you tested> -t llm-vllm:1.0 .
# There is no default on purpose. Do not use "latest" in production.
ARG VLLM_VERSION
# Bake the model into the image so pods start fast.
# Check how the base image starts the server: docker inspect <image> (look at Entrypoint),
# then make the CMD below match it. Keep the model name in CMD equal to the MODEL arg.
# The model name appears twice (ARG MODEL and CMD): change both together.
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV HF_HOME=/models
ARG MODEL=Qwen/Qwen2.5-1.5B-Instruct

# download the weights at build time
# For a gated model, build with: docker build --secret id=hf,env=HF_TOKEN .
# The token is then read only during this step and is not kept in the image.
# (huggingface_hub is already inside the vLLM image, so nothing to install.)
RUN --mount=type=secret,id=hf,required=false \
    HF_TOKEN="$(cat /run/secrets/hf 2>/dev/null || true)" python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

# the weights are in the image now: never go to the internet to look for them.
# (Set this AFTER the download step above. Remove it if you load a model at run time.)
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
# the base image entrypoint is the OpenAI server; we pass the flags
CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--port", "8000", "--gpu-memory-utilization", "0.90", "--max-model-len", "8192", "--shutdown-timeout", "100"]
# --shutdown-timeout: wait up to 100 s for running requests on stop. Keep it plus the
# preStop sleep below terminationGracePeriodSeconds (k8s.yaml). If your vLLM version
# does not have this flag, remove it.
