"""Run the HELM x CALM suite on a Modal GPU.

Written against the installed client (modal 1.5.5) rather than the documentation,
because ``modal.com`` is blocked by this environment's egress proxy -- the same
policy that blocks ``api.modal.com``, so **this app cannot be launched from the
session that wrote it**. It is here to run unchanged from anywhere that can
reach Modal.

    modal token set --token-id ak-... --token-secret as-... --profile=NAME
    modal profile activate NAME
    modal run CALM/suite/modal_app.py::tier2          # the first tier worth a GPU
    modal run CALM/suite/modal_app.py::gate           # reproduction gate only
    modal run CALM/suite/modal_app.py::fetch_wikitext103

**Why the GPU changes what is measurable.** Everything measured locally sits at
~450K parameters on WikiText-2, which is well below the scale at which
architecture comparisons are known to predict anything. Two things become
available here and nowhere else:

* **WikiText-103** -- the corpus HELM and CALM both actually train on. It exceeds
  raw GitHub's file limit, so the local session cannot fetch it; a Modal
  container has ordinary network access and can. :func:`fetch_wikitext103` puts
  it on a Volume once, and every later run reuses it.
* **Tier 2 and 3 shapes** -- 12 layers at width 513, and HELM's published 120M
  setting. Tier 3 is the only tier whose numbers are comparable to the paper
  tables.

**Cost control.** ``timeout`` is set per function and ``max_containers=1`` keeps
a mistake from fanning out across GPUs. Tier 2 with three seeds is roughly two
GPU-hours on an A10G; tier 3 is a different order of magnitude and is left
without a convenience entry point on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch==2.5.1",
        "geoopt",
        "numpy",
        "transformers",
        # torch_geometric is imported by parts of hypercore; the sparse extras
        # are not needed for the modules this suite touches.
        "torch_geometric",
    )
    # Ship the repository itself. add_local_dir runs last so code changes do not
    # invalidate the (slow) pip layer above.
    .add_local_dir(REPO, remote_path="/root/HEM",
                   ignore=modal.FilePatternMatcher(
                       "**/.git/**", "**/__pycache__/**", "**/*.pyc",
                       "**/CALM/data/*.txt"))
)

app = modal.App("helm-calm-suite", image=image)
data = modal.Volume.from_name("helm-calm-data", create_if_missing=True)
results = modal.Volume.from_name("helm-calm-results", create_if_missing=True)

DATA_DIR = "/data"
RESULTS_DIR = "/results"


def _link_corpora() -> None:
    """Point ``CALM/data`` at the Volume so corpus.py finds the files."""
    import os
    target = Path("/root/HEM/CALM/data")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if not target.is_symlink():
            for item in target.glob("*.txt"):
                item.unlink()
            target.rmdir()
        else:
            target.unlink()
    os.symlink(DATA_DIR, target)


@app.function(volumes={DATA_DIR: data}, timeout=60 * 30)
def fetch_corpora(include_103: bool = True) -> dict:
    """Put the corpora on the Volume once. Run before any training function.

    WikiText-2 and PTB come from the same raw.githubusercontent URLs the local
    session uses, so the bytes are identical and the digests printed by
    ``corpus.py`` should match. WikiText-103 is the addition a GPU environment
    makes possible.
    """
    import subprocess
    import urllib.request
    import zipfile

    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}

    base2 = ("https://raw.githubusercontent.com/pytorch/examples/main/"
             "word_language_model/data/wikitext-2")
    base_ptb = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
    for split in ("train", "valid", "test"):
        for name, url in (("wikitext2", f"{base2}/{split}.txt"),
                          ("ptb", f"{base_ptb}/ptb.{split}.txt")):
            path = out / f"{name}.{split}.txt"
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            sizes[path.name] = path.stat().st_size

    if include_103:
        # The reason this function exists on Modal rather than locally.
        archive = out / "wikitext-103-raw-v1.zip"
        if not (out / "wikitext103.train.txt").exists():
            url = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
                   "main/wikitext-103-raw-v1/train-00000-of-00002.parquet")
            try:
                urllib.request.urlretrieve(
                    "https://wikitext.smerity.com/wikitext-103-raw-v1.zip", archive)
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(out)
                mapping = {"wiki.train.raw": "wikitext103.train.txt",
                           "wiki.valid.raw": "wikitext103.valid.txt",
                           "wiki.test.raw": "wikitext103.test.txt"}
                for src, dst in mapping.items():
                    found = next(out.rglob(src), None)
                    if found:
                        found.rename(out / dst)
                        sizes[dst] = (out / dst).stat().st_size
                archive.unlink(missing_ok=True)
            except Exception as error:            # noqa: BLE001
                # Reported, not swallowed: the run can still proceed on
                # WikiText-2, but the caller must know it is not on 103.
                sizes["wikitext103_error"] = repr(error)[:300]

    data.commit()
    return sizes


def _run(argv: list[str]) -> str:
    """Invoke the suite exactly as the CLI would, and return its output."""
    import subprocess
    _link_corpora()
    command = [sys.executable, "CALM/suite/run.py", *argv]
    completed = subprocess.run(command, cwd="/root/HEM", text=True,
                               capture_output=True,
                               env={"PYTHONPATH": "/root/HEM", "PATH": "/usr/bin:/bin",
                                    "HOME": "/root"})
    print(completed.stdout)
    if completed.returncode:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"suite exited {completed.returncode}")
    return completed.stdout


@app.function(gpu=GPU, volumes={DATA_DIR: data, RESULTS_DIR: results},
              timeout=60 * 60 * 2, max_containers=1)
def gate(corpus: str = "wikitext2", steps: int = 40000, seeds: str = "0,1,2") -> str:
    """The reproduction gate, on a GPU: does HELM beat matched Euclidean at all?

    Run this before ``tier2``. If the geometry effect is absent in the discrete
    column, the CALM column cannot be attributed to geometry and the suite will
    refuse to report an interaction -- spending the rest of the GPU budget on it
    would buy a number that does not mean what it appears to.
    """
    out = _run(["--tier", "2", "--steps", str(steps), "--seeds", seeds,
                "--corpus", corpus, "--device", "cuda",
                "--cells", "helm_discrete,euclid_discrete",
                "--out", f"{RESULTS_DIR}/gate_{corpus}.json"])
    results.commit()
    return out


@app.function(gpu=GPU, volumes={DATA_DIR: data, RESULTS_DIR: results},
              timeout=60 * 60 * 6, max_containers=1)
def tier2(corpus: str = "wikitext2", seeds: str = "0,1,2",
          steps: int = 0, patch: int = 4) -> str:
    """The full 2x2 at tier-2 shape. Roughly two GPU-hours at three seeds."""
    argv = ["--tier", "2", "--seeds", seeds, "--corpus", corpus,
            "--patch", str(patch), "--device", "cuda",
            "--out", f"{RESULTS_DIR}/tier2_{corpus}.json"]
    if steps:
        argv += ["--steps", str(steps)]
    out = _run(argv)
    results.commit()
    return out


@app.function(volumes={RESULTS_DIR: results}, timeout=60 * 10)
def collect() -> dict:
    """Return the result JSONs so they can be pulled back for analysis."""
    import json
    out = {}
    for path in Path(RESULTS_DIR).glob("*.json"):
        out[path.name] = json.loads(path.read_text())
    return out


@app.local_entrypoint()
def main(what: str = "gate", corpus: str = "wikitext2", seeds: str = "0,1,2"):
    """``modal run CALM/suite/modal_app.py --what gate|tier2|fetch``."""
    if what == "fetch":
        print(fetch_corpora.remote())
        return
    print(fetch_corpora.remote())
    print((gate if what == "gate" else tier2).remote(corpus=corpus, seeds=seeds))
