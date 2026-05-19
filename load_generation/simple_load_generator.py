import multiprocessing
import os
import random
import subprocess
import time
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler

# === Configuration ===
SEED = 42
MIN_PROCESSES = 15
MAX_PROCESSES = 20
MIN_DURATION = 15
MAX_DURATION = 30
RUNTIME_LIMIT = None  # e.g., 600 for 10 minutes or None for indefinite

random.seed(SEED)
running = True


# === Custom HTTP Request Handler to mute output of server ===
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<html><body><p>OK</p></body></html>')
    def log_message(self, format, *args):
        return

# === Graceful Shutdown ===
def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping simulation...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# === Workload Definitions ===

def cpu_load(duration, intensity):
    delay = max(0.00001, 0.01 * (1.0 - intensity))
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = [x**2 for x in range(100000)]
        time.sleep(delay)

def memory_load(duration, intensity):
    max_chunks = int(50 + 200 * intensity)
    big_list = []
    end_time = time.time() + duration
    while time.time() < end_time:
        big_list.append(os.urandom(1024 * 1024))
        if len(big_list) > max_chunks:
            big_list.pop(0)
        time.sleep(0.05)

def io_load(duration, intensity):
    block_size = int(1024 * 1024 * intensity)
    end_time = time.time() + duration
    with open(f"/tmp/io_temp_{os.getpid()}.bin", "wb") as f:
        while time.time() < end_time:
            f.write(os.urandom(block_size))
            f.flush()
            time.sleep(0.1 * (1.0 - intensity))

def web_server_load(duration, intensity):
    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Hello!")

    port = 8000 + os.getpid() % 1000
    server = HTTPServer(("localhost", port), MyHandler)
    server_proc = multiprocessing.Process(target=server.serve_forever)
    server_proc.daemon = True
    server_proc.start()

    # Simulated traffic to local server
    end_time = time.time() + duration
    interval = max(0.05, 1.0 - intensity)
    while time.time() < end_time:
        try:
            subprocess.run(["curl", "-s", f"http://localhost:{port}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
        time.sleep(interval)

    server.shutdown()
    server.server_close()

# === Workload Runner ===

WORKLOADS = {
    "cpu": cpu_load,
    "memory": memory_load,
    "io": io_load,
    "web": web_server_load
}

def run_random_workload():
    workload = random.choice(list(WORKLOADS.keys()))
    duration = random.randint(MIN_DURATION, MAX_DURATION)
    intensity = round(random.uniform(0.2, 1.0), 2)
    pid = os.getpid()

    print(f"[+] PID {pid} → {workload.upper()} | Duration: {duration}s | Intensity: {intensity}")
    start_time = time.time()

    try:
        WORKLOADS[workload](duration, intensity)
    except Exception as e:
        print(str(e.__traceback__))
        print(f"[!] PID {pid} error: {e}")

    print(f"[✓] PID {pid} → {workload.upper()} finished ({int(time.time() - start_time)}s)")

# === Simulation Loop ===

def spawn_random_processes():
    print(f"[~] Starting simulation (random process count, seed={SEED})")
    start_time = time.time()

    while running:
        proc_count = random.randint(MIN_PROCESSES, MAX_PROCESSES)
        print(f"\n[#] Spawning {proc_count} processes this round")
        processes = []

        for _ in range(proc_count):
            p = multiprocessing.Process(target=run_random_workload)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if RUNTIME_LIMIT and (time.time() - start_time) > RUNTIME_LIMIT:
            break

    print("[✔] Simulation ended.")

# === Entry Point ===

if __name__ == "__main__":
    spawn_random_processes()
