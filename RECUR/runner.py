"""A job is a dictionary; a run is a job executed in its own process.

Every experiment here is a list of small training runs that differ in a few
config fields. Describing a run as plain data rather than as a closure buys
three things that matter more than elegance:

* it is picklable, so the four cores can be used as four single-threaded
  workers -- measured at ~1.5x the throughput of one four-threaded process on
  this machine, because these models are too small to keep four threads busy;
* it is serialisable, so the exact job that produced a result file is stored
  inside that result file;
* it is inspectable, so ``--dry-run`` prints the whole experiment, including
  the fields each arm differs in, before spending an hour on it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def execute(job: Dict) -> Dict:
    """Run one job in this process. Imports are inside for spawn-safety."""
    import torch
    torch.set_num_threads(int(job.get("threads", 1)))

    from harness import train_bytes, train_hops, save
    from model import Config
    from tasks import HopSpec, load_bytes

    task = job["task"]
    overrides = dict(job.get("cfg", {}))
    overrides["seed"] = job["seed"]

    if task in ("hops", "twochain"):
        spec = HopSpec(two_chain=(task == "twochain"), **job.get("spec", {}))
        cfg = Config(vocab_size=spec.vocab_size, max_seq_len=spec.seq_len,
                     **overrides)
        result, _ = train_hops(cfg, spec, steps=job["steps"],
                               batch_size=job.get("batch", 32),
                               lr=job.get("lr", 2e-3),
                               data_seed=job.get("data_seed", 1234),
                               eval_every=job.get("eval_every", 0),
                               eval_loops=tuple(job.get("eval_loops", ())))
    elif task.startswith("bytes"):
        corpus = load_bytes(task.split(":")[1] if ":" in task else "wikitext2")
        cfg = Config(vocab_size=256, max_seq_len=job.get("seq_len", 128),
                     **overrides)
        result, _ = train_bytes(cfg, corpus, steps=job["steps"],
                                seq_len=job.get("seq_len", 128),
                                batch_size=job.get("batch", 16),
                                lr=job.get("lr", 1.5e-3),
                                data_seed=job.get("data_seed", 1234))
    else:
        raise ValueError(f"unknown task {task!r}")

    result["job"] = job
    save(job["name"], result)
    return result


def run_jobs(jobs: List[Dict], workers: int = 4, dry_run: bool = False,
             skip_done: bool = True) -> List[Dict]:
    """Execute jobs in parallel, skipping any whose result file already exists.

    Skipping is on by default so that an interrupted experiment resumes instead
    of re-spending the compute; ``--force`` in the experiment scripts turns it
    off. The skip is by job *name*, and names encode every field that varies,
    so a changed arm gets a new name rather than silently reusing an old file.
    """
    from harness import RESULTS
    pending = []
    for job in jobs:
        path = RESULTS / f"{job['name']}.json"
        if skip_done and path.exists():
            continue
        pending.append(job)

    print(f"{len(jobs)} jobs, {len(pending)} to run, {workers} workers")
    for job in pending:
        print("  ", job["name"], json.dumps(job.get("cfg", {}), sort_keys=True))
    if dry_run:
        return []

    done, t0 = [], time.time()
    if workers <= 1:
        for job in pending:
            done.append(execute(job))
            print(f"  done {job['name']} ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, job): job for job in pending}
            for fut in futures:
                pass
            for fut, job in futures.items():
                result = fut.result()
                done.append(result)
                print(f"  done {job['name']} "
                      f"({result['seconds']:.0f}s, {time.time() - t0:.0f}s elapsed)",
                      flush=True)
    return done


def load_results(prefix: str) -> Dict[str, Dict]:
    from harness import RESULTS
    out = {}
    for path in sorted(RESULTS.glob(f"{prefix}*.json")):
        out[path.stem] = json.loads(path.read_text())
    return out


def mean_sd(values):
    import statistics
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)
