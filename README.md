# vLLM & SGLang: From First Request to Production (code)

Companion code for the book. The book itself is sold on Amazon Kindle.

- `code/demos/` small Python demos and simulators (no GPU needed for the `sim_*.py` files)
- `code/deploy/` benchmark sweep, cost calculator, Dockerfile, Kubernetes and Prometheus files

Run the simulators from `code/demos` (`pip install matplotlib`). The simulators save their charts as PNG files in the same folder.

Test everything without a GPU: `pip install matplotlib httpx`, then `python code/tests/test_smoke.py`. A CI job runs it on Python 3.10, 3.11 and 3.12.

Licence: MIT. Report problems in the Issues tab.
