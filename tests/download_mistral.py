"""Parallel range downloader for ModelScope shards."""
import concurrent.futures as cf
import os
import sys
import time
import urllib.request

BASE = "https://modelscope.cn/models/mistralai/Mistral-Small-3.1-24B-Instruct-2503/resolve/master"
DST = r"C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503"
SHARDS = [f"model-{i:05d}-of-00010.safetensors" for i in range(1, 11)]
CHUNK = 32 * 1024 * 1024
WORKERS = 24


def size_of(name):
    req = urllib.request.Request(
        f"{BASE}/{name}", headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.rsplit("/", 1)[-1])
        return int(r.headers["Content-Length"])


def fetch(job):
    name, idx, start, end = job
    part = os.path.join(DST, f"{name}.part{idx}")
    want = end - start + 1
    for attempt in range(6):
        try:
            if os.path.exists(part) and os.path.getsize(part) == want:
                return want
            req = urllib.request.Request(
                f"{BASE}/{name}", headers={"Range": f"bytes={start}-{end}"})
            got = 0
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while True:
                    buf = r.read(1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
            if got == want:
                return got
        except Exception as exc:  # noqa: BLE001
            print(f"retry {name} part{idx}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed {name} part{idx}")


def main():
    jobs = []
    totals = {}
    for name in SHARDS:
        final = os.path.join(DST, name)
        n = size_of(name)
        totals[name] = n
        if os.path.exists(final) and os.path.getsize(final) == n:
            print(f"skip done {name}", flush=True)
            continue
        parts = (n + CHUNK - 1) // CHUNK
        for i in range(parts):
            jobs.append((name, i, i * CHUNK, min(n - 1, (i + 1) * CHUNK - 1)))
    print(f"jobs={len(jobs)}", flush=True)
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for got in ex.map(fetch, jobs):
            done += got
            if done % (512 * 1024 * 1024) < CHUNK:
                el = time.time() - t0
                print(f"progress {done/1e9:.2f} GB {done/el/1e6:.1f} MB/s", flush=True)
    for name, n in totals.items():
        final = os.path.join(DST, name)
        if os.path.exists(final) and os.path.getsize(final) == n:
            continue
        parts = (n + CHUNK - 1) // CHUNK
        with open(final, "wb") as out:
            for i in range(parts):
                part = os.path.join(DST, f"{name}.part{i}")
                with open(part, "rb") as f:
                    while True:
                        buf = f.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                os.remove(part)
        assert os.path.getsize(final) == n, name
        print(f"assembled {name} {n/1e9:.2f} GB", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
