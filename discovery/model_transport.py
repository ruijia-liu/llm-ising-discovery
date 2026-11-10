"""No-shell JSONL transport with bounded liveness checks, never timeout replay."""
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import time


class ModelWorker:
    def __init__(self, model, adapter=None, *, log_path, timeout=30):
        self.timeout = timeout
        self.stderr = Path(log_path).open('a', encoding='utf-8')
        self.stderr.write(json.dumps({'worker_invocation_started_ns': time.time_ns()})+'\n')
        self.stderr.flush()
        args = [sys.executable, '-u', str(Path('student_inference_worker.py').resolve()), '--model', str(Path(model).resolve())]
        if adapter:
            args += ['--adapter', str(Path(adapter).resolve())]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, encoding='utf-8', bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.queue = Queue()
        def read():
            for line in self.process.stdout:
                self.queue.put(line)
            self.queue.put(None)
        Thread(target=read, daemon=True).start()
        try:
            self.ready = self.receive()
            if self.ready.get('ready') is not True:
                raise RuntimeError('Inference worker did not initialize.')
        except Exception:
            self.close()
            raise

    def receive(self):
        while True:
            try:
                line = self.queue.get(timeout=min(self.timeout, 60))
                break
            except Empty:
                # An observation timeout is not completion. Continue waiting on
                # this exact request while the authoritative process is alive.
                if self.process.poll() is not None:
                    raise RuntimeError(f'Inference worker exited (code {self.process.returncode}).')
        if line is None:
            raise RuntimeError(f'Inference worker exited (code {self.process.poll()}).')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError('Inference worker: '+reply['error'])
        return reply

    def __call__(self, messages, max_new_tokens):
        self.process.stdin.write(json.dumps({'messages': messages, 'max_new_tokens': max_new_tokens}, ensure_ascii=True)+'\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
