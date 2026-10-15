"""Smoke test for every script in code/. Run: python code/tests/test_smoke.py
No GPU needed. A tiny fake server stands in for vLLM and SGLang."""
import json, os, py_compile, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS, DEPLOY = os.path.join(CODE, "demos"), os.path.join(CODE, "deploy")
ENV = dict(os.environ, MPLBACKEND="Agg")


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ["Hello", " there", " friend"]:
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"choices": [], "usage": {"completion_tokens": 3}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, f"{cmd}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def main():
    for name in ["vllm_app.py", "sglang_program.py"]:
        py_compile.compile(os.path.join(DEMOS, name), doraise=True)
    py_compile.compile(os.path.join(DEPLOY, "plot_sweep.py"), doraise=True)
    run(["bash", "-n", "run_sweep.sh"], DEPLOY)

    out = run([sys.executable, "cost.py", "1.20", "2400"], DEPLOY)
    assert "$0.139" in out, out

    with tempfile.TemporaryDirectory() as tmp:
        for sim in ["sim_memory.py", "sim_kv.py", "sim_radix.py", "sim_batching.py"]:
            run([sys.executable, os.path.join(DEMOS, sim)], tmp)
        assert any(f.endswith(".png") for f in os.listdir(tmp))

        server = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        csv_path = os.path.join(tmp, "results.csv")
        out = run([sys.executable, os.path.join(DEMOS, "bench.py"),
                   "--url", url, "--model", "fake", "--concurrency", "4",
                   "--num-requests", "8", "--csv", csv_path, "--engine", "fake"], tmp)
        assert "requests ok     : 8/8" in out, out
        run([sys.executable, os.path.join(DEPLOY, "plot_sweep.py"), csv_path], tmp)
        server.shutdown()
    print("all smoke tests passed on Python", sys.version.split()[0])


if __name__ == "__main__":
    main()
