"""gdrive.py - incremental sync of a Google Drive folder into cfg.root using the Drive v3 REST API and an API key.

SECURITY
  * The API key is read from the environment / Kaggle Secrets (utils.get_secret) and sent ONLY in the `X-Goog-Api-Key` header,
    never in a URL, so it cannot leak through logs, tracebacks or `requests` error messages. It is also scrubbed from any message we raise.
  * Nothing about the key is written to disk (the manifest / report contain file metadata only).

LIMITS (important)
  * An API key can only read content that is shared "Anyone with the link" (or public). For a private folder Google returns 404/403 and
    you need OAuth instead - this module detects that and says so.

BEHAVIOUR
  * Lists the folder tree in parallel, caches the listing (cache/gdrive_manifest.json), selects only the files the pipeline reads
    (profile="pipeline") or everything (profile="all"), skips files that are already complete (size match), downloads the rest in
    parallel to *.part files with md5 verification, retries on 429 / 5xx, and writes cache/download_report.json.
"""
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from utils import LOG, get_secret, tq

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
GOOGLE_NATIVE = "application/vnd.google-apps."
SMALL_EXT = {".csv", ".tsv", ".txt", ".json", ".md", ".yml", ".yaml", ".xlsx", ".xls", ".xml", ".nxml", ".soft", ".cppipe", ".pattern", ".screen", ".py", ".sh"}
MB = 1024 * 1024


class DriveError(RuntimeError):
    pass


class DriveAccessError(DriveError):
    pass


def parse_folder_id(s):
    s = str(s).strip()
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", s) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    raise DriveError(f"cannot read a Drive folder id from {s!r}")


def _safe_name(n):
    return re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", n).strip() or "_"


class DriveClient:
    def __init__(self, api_key, api_base="https://www.googleapis.com/drive/v3", retries=7):
        if not api_key:
            raise DriveError("empty API key")
        self._key, self.base, self.retries = api_key, api_base.rstrip("/"), retries
        self._tl = threading.local()
        self._penalty_until, self._plock, self.rate_limited = 0.0, threading.Lock(), 0

    # ------------------------------------------------------------------ plumbing
    def _scrub(self, text):
        return str(text).replace(self._key, "***")

    def _sess(self, with_key=True):
        name = "s_key" if with_key else "s_plain"
        s = getattr(self._tl, name, None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
            s.mount("https://", adapter); s.mount("http://", adapter)
            if with_key:
                s.headers["X-Goog-Api-Key"] = self._key
            setattr(self._tl, name, s)
        return s

    @staticmethod
    def _reason(r):
        try:
            j = r.json()["error"]
            return (j.get("errors") or [{}])[0].get("reason", ""), j.get("message", "")
        except Exception:
            return "", (r.text or "")[:200]

    def _request(self, url, params=None, stream=False, with_key=True):
        last = None
        for attempt in range(self.retries):
            wait = self._penalty_until - time.time()
            if wait > 0:                                  # a rate limit was hit by ANY thread: everybody backs off together
                time.sleep(min(wait, 30))
            try:
                r = self._sess(with_key).get(url, params=params, stream=stream, timeout=(15, 120))
            except requests.RequestException as e:
                last = self._scrub(repr(e))[:200]
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                continue
            if r.status_code == 200:
                return r
            reason, msg = self._reason(r)
            retriable = r.status_code in (429, 500, 502, 503, 504) or (r.status_code == 403 and "ratelimit" in reason.lower().replace("_", ""))
            if retriable:
                last = f"HTTP {r.status_code} {reason}"
                r.close()
                back = min(60, 2 ** attempt) * (0.5 + random.random())
                if r.status_code in (403, 429):
                    with self._plock:
                        self.rate_limited += 1
                        self._penalty_until = max(self._penalty_until, time.time() + back)
                else:
                    time.sleep(back)
                continue
            r.close()
            raise DriveAccessError(self._scrub(f"HTTP {r.status_code} ({reason or 'error'}): {msg}"))
        raise DriveError(f"giving up after {self.retries} attempts: {last}")

    # ------------------------------------------------------------------ API calls
    def get_meta(self, fid):
        r = self._request(f"{self.base}/files/{fid}", params={"fields": "id,name,mimeType,size,md5Checksum", "supportsAllDrives": "true"})
        return r.json()

    def list_children(self, fid):
        out, token = [], None
        while True:
            params = {"q": f"'{fid}' in parents and trashed=false", "pageSize": 1000, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
                      "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum,shortcutDetails)"}
            if token:
                params["pageToken"] = token
            j = self._request(f"{self.base}/files", params=params).json()
            out += j.get("files", [])
            token = j.get("nextPageToken")
            if not token:
                return out

    def walk(self, root_id, workers=16):
        """Breadth-first parallel listing. -> list of dict(id, name, rel, size, md5, mime)"""
        files, level, n_dirs = [], [(root_id, "")], 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            while level:
                results = list(ex.map(lambda t: (t[1], self.list_children(t[0])), level))
                nxt = []
                for rel, kids in results:
                    used = set()
                    for k in kids:
                        mime, kid, name = k.get("mimeType", ""), k["id"], _safe_name(k["name"])
                        if mime == SHORTCUT:
                            sd = k.get("shortcutDetails") or {}
                            mime, kid = sd.get("targetMimeType", ""), sd.get("targetId", kid)
                        if name in used:                                   # Drive allows duplicate names in one folder
                            stem, ext = os.path.splitext(name); name = f"{stem}__{kid[:6]}{ext}"
                        used.add(name)
                        path = f"{rel}/{name}" if rel else name
                        if mime == FOLDER:
                            nxt.append((kid, path)); n_dirs += 1
                        else:
                            files.append(dict(id=kid, name=name, rel=path, size=int(k["size"]) if k.get("size") else None, md5=k.get("md5Checksum"), mime=mime))
                level = nxt
                LOG.info(f"[gdrive] listing: {len(files):,} files, {n_dirs:,} folders so far ({time.time()-t0:.0f}s), next level {len(level)} folders")
        return files

    # ------------------------------------------------------------------ download
    def download(self, f, dest, progress=None):
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        errs = []
        for route in ("api", "public"):
            try:
                if route == "api":
                    r = self._request(f"{self.base}/files/{f['id']}", params={"alt": "media", "supportsAllDrives": "true"}, stream=True)
                else:   # fallback for files whose API download quota is exhausted (works for link-shared files only)
                    r = self._request("https://drive.usercontent.google.com/download", params={"id": f["id"], "export": "download", "confirm": "t"}, stream=True, with_key=False)
                    if "text/html" in r.headers.get("Content-Type", ""):
                        r.close(); raise DriveAccessError("public endpoint returned an HTML page")
                md5, n = hashlib.md5(), 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk); md5.update(chunk); n += len(chunk)
                            if progress:
                                progress(len(chunk))
                r.close()
                if f.get("size") is not None and n != f["size"]:
                    raise DriveError(f"size mismatch {n} != {f['size']}")
                if f.get("md5") and md5.hexdigest() != f["md5"]:
                    raise DriveError("md5 mismatch")
                os.replace(part, dest)
                return n
            except DriveAccessError as e:
                errs.append(f"{route}: {e}")
                if "downloadQuota" not in str(e) and "cannotDownload" not in str(e) and route == "api" and "HTTP 403" not in str(e):
                    break
            except Exception as e:  # noqa: BLE001
                errs.append(f"{route}: {self._scrub(repr(e))[:150]}")
                break
            finally:
                if part.exists() and not dest.exists():
                    try:
                        part.unlink()
                    except OSError:
                        pass
        raise DriveError("; ".join(errs))


# ======================================================================= file selection
def _ctc_parts(rel):
    parts = rel.replace("\\", "/").split("/")
    if "training" not in parts:
        return None
    i = parts.index("training")
    return parts, i


def select_files(files, profile="pipeline", extra=()):
    """Choose the files the pipeline stages actually read. 'all' keeps everything.
    pipeline profile:
      * every small table / text / xml / json (<= 60 MB) in any dataset
      * jump_cell_painting: *.parquet (profiles)         allen_cell_explorer: *.tif (OME-TIFF volumes)
      * bbbc/BBBC038: stage1_train images + masks         bbbc/BBBC021: metadata + DAPI (w1) images if present
      * cell_tracking_challenge: ground-truth SEG masks + ONLY the raw frames that have a ground-truth mask
      * pmc_open_access / livertox: *.xml / *.nxml (full text for literature retrieval; PDFs and JPGs are skipped)
    """
    extra_res = [re.compile(p) for p in extra]
    if profile == "all":
        return [f for f in files if not f["mime"].startswith(GOOGLE_NATIVE)]
    gt_keys = set()
    for f in files:                                                    # pass 1: which CTC frames have a manual segmentation?
        p = _ctc_parts(f["rel"])
        if p and f["name"].startswith("man_seg") and f["rel"].replace("\\", "/").split("/")[-2] == "SEG":
            parts, i = p
            m = re.fullmatch(r"(\d+)_GT", parts[-3])
            if m and len(parts) > i + 1:
                gt_keys.add((parts[i + 1], m.group(1), re.findall(r"\d+", f["name"])[0]))
    out = []
    for f in files:
        if f["mime"].startswith(GOOGLE_NATIVE):
            continue
        rel = f["rel"].replace("\\", "/"); parts = rel.split("/"); top = parts[0].lower(); name = f["name"]; ext = os.path.splitext(name)[1].lower()
        size = f["size"] or 0
        small = ext in SMALL_EXT and size <= 60 * MB
        keep = small
        if top == "jump_cell_painting":
            keep = keep or ext == ".parquet"
        elif top == "allen_cell_explorer":
            keep = keep or ext in (".tif", ".tiff")
        elif top == "bbbc":
            if "BBBC038" in parts[:3] and "stage1_train" in parts:
                keep = keep or ext == ".png"
            elif "BBBC021" in parts[:3]:
                keep = keep or (ext in (".tif", ".tiff") and "_w1" in name)
        elif top == "cell_tracking_challenge":
            p = _ctc_parts(rel)
            if p:
                pp, i = p
                if name.startswith("man_seg") and pp[-2] == "SEG":
                    keep = True
                elif re.match(r"^t\d+\.tif", name) and len(pp) > i + 2 and re.fullmatch(r"\d+", pp[-2]) and (pp[i + 1], pp[-2], re.findall(r"\d+", name)[0]) in gt_keys:
                    keep = True
        elif top in ("pmc_open_access", "livertox"):
            keep = keep or ext in (".xml", ".nxml")
        if not keep and extra_res:
            keep = any(r.search(rel) for r in extra_res)
        if keep:
            out.append(f)
    return out


# ======================================================================= sync
def _strip_dataset_root(files):
    """If the shared folder CONTAINS a folder named open_datasets, treat that as the dataset root (and ignore siblings)."""
    tops = {f["rel"].split("/")[0] for f in files if "/" in f["rel"]}
    cand = [t for t in tops if t.lower() == "open_datasets"]
    if cand:
        pre = cand[0] + "/"
        LOG.info(f"[gdrive] using sub-folder '{cand[0]}' as the dataset root")
        return [dict(f, rel=f["rel"][len(pre):]) for f in files if f["rel"].startswith(pre)]
    return files


_BG = {"thread": None, "result": None, "error": None}


def _download_batch(cli, root, items, cfg, label):
    """Download `items` in parallel. A monitor thread prints a progress line every 20 s (progress bars do not render in Kaggle logs).
    -> (failed dict rel->error, bytes_done)"""
    if not items:
        return {}, 0
    need = sum(f["size"] or 0 for f in items)
    st = dict(files=0, bytes=0, failed=0)
    lock, stop = threading.Lock(), threading.Event()
    t0 = time.time()

    def prog(n):
        with lock:
            st["bytes"] += n

    def one(f):
        try:
            cli.download(f, root / f["rel"], prog)
            err = None
        except Exception as e:  # noqa: BLE001
            err = f"{f['rel']}: {cli._scrub(e)[:200]}"
        with lock:
            st["files"] += 1
            st["failed"] += err is not None
        return err

    def monitor():
        while not stop.wait(20):
            with lock:
                fd, by, fl = st["files"], st["bytes"], st["failed"]
            el = max(time.time() - t0, 1e-6)
            rate = by / el
            eta = (need - by) / rate / 60 if rate > 0 and need else float("nan")
            LOG.info(f"[gdrive] {label}: {fd:,}/{len(items):,} files, {by/1e9:.2f}/{need/1e9:.2f} GB, {rate/1e6:.1f} MB/s, {fd/el:.0f} files/s, "
                     f"failed {fl}, rate-limit hits {cli.rate_limited}, ETA {eta:.0f} min")
    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    failed = {}
    pending = list(items)
    try:
        for rnd in range(3):                                         # up to 3 passes: whatever failed is retried
            with ThreadPoolExecutor(max_workers=cfg.download_workers) as ex:
                res = list(ex.map(one, pending))
            bad = [(f, r) for f, r in zip(pending, res) if r]
            pending = [f for f, _ in bad]
            failed = {f["rel"]: r for f, r in bad}
            if not pending:
                break
            LOG.warning(f"[gdrive] {label}: {len(pending)} downloads failed (pass {rnd+1}/3); retrying in 10 s. First error: {bad[0][1][:160]}")
            with lock:
                st["files"] -= len(pending); st["failed"] -= len(pending)
            time.sleep(10)
    finally:
        stop.set()
    el = time.time() - t0
    LOG.info(f"[gdrive] {label} done: {len(items)-len(failed):,}/{len(items):,} files, {st['bytes']/1e9:.2f} GB in {el/60:.1f} min ({st['bytes']/max(el,1e-6)/1e6:.1f} MB/s), failed {len(failed)}")
    return failed, st["bytes"]


def _split_pending(root, files):
    pending, cached = [], 0
    for f in files:
        dest = root / f["rel"]
        if dest.exists() and (f["size"] is None or dest.stat().st_size == f["size"]):
            cached += 1
        else:
            pending.append(f)
    return pending, cached


def _check_disk(root, pending, what):
    need = sum(f["size"] or 0 for f in pending)
    free = shutil.disk_usage(root).free
    LOG.info(f"[gdrive] {what}: {len(pending):,} files / {need/1e9:.2f} GB to download; free disk {free/1e9:.1f} GB")
    if need * 1.05 + 1.5e9 > free:
        raise DriveError(f"not enough disk space for {what}: need {need/1e9:.1f} GB (+1.5 GB headroom for caches/models), free {free/1e9:.1f} GB.")
    return need


def sync_dataset(cfg, refresh_manifest=False, verbose=True):
    """Make cfg.root contain the Drive folder. Two phases when cfg.download_profile == 'all':
         phase 1 (blocking)   : every file the pipeline stages read (~4.9 GB)   -> the pipeline can start
         phase 2 (background) : the remaining files (~3.9 GB: PDFs, unused images, ...) while the stages run; call wait_background() at the end
    Returns the phase-1 report (also saved to cache/download_report.json)."""
    key = get_secret(cfg.gdrive_secret)
    if not key:
        raise DriveError(f"Drive API key not found. Add a Kaggle Secret named {cfg.gdrive_secret!r} (Add-ons -> Secrets) AND tick it for this notebook, "
                         f"or set the environment variable {cfg.gdrive_secret}. (The key itself is never printed.)")
    fid = parse_folder_id(cfg.gdrive_folder)
    cli = DriveClient(key, cfg.gdrive_api_base)
    root = Path(cfg.root); root.mkdir(parents=True, exist_ok=True)
    manifest = cfg.cache / "gdrive_manifest.json"
    files = None
    if manifest.exists() and not refresh_manifest:
        try:
            m = json.loads(manifest.read_text())
            if m.get("folder") == fid and m.get("api_base") == cfg.gdrive_api_base:
                files = m["files"]; LOG.info(f"[gdrive] using cached listing ({len(files):,} files). run_pipeline(refresh_manifest=True) re-lists the folder.")
        except Exception:
            files = None
    if files is None:
        try:
            meta = cli.get_meta(fid)
        except DriveAccessError as e:
            raise DriveAccessError(f"cannot open the Drive folder with this API key: {e}\n"
                                   "An API key can only read folders shared as 'Anyone with the link -> Viewer'. If the folder is private, "
                                   "change its sharing setting (or use OAuth).") from None
        if meta.get("mimeType") != FOLDER:
            raise DriveError(f"{fid} is not a folder (mimeType={meta.get('mimeType')})")
        LOG.info(f"[gdrive] listing folder '{meta.get('name')}' ...")
        files = _strip_dataset_root(cli.walk(fid, cfg.download_workers))
        manifest.write_text(json.dumps(dict(folder=fid, api_base=cfg.gdrive_api_base, files=files)))
    total = sum(f["size"] or 0 for f in files)
    LOG.info(f"[gdrive] folder holds {len(files):,} files, {total/1e9:.2f} GB")
    core = select_files(files, "pipeline", cfg.download_extra)
    wanted = select_files(files, cfg.download_profile, cfg.download_extra) if cfg.download_profile != "pipeline" else core
    core_rel = {f["rel"] for f in core}
    rest = [f for f in wanted if f["rel"] not in core_rel]
    LOG.info(f"[gdrive] profile '{cfg.download_profile}': phase 1 = {len(core):,} files / {sum(f['size'] or 0 for f in core)/1e9:.2f} GB the pipeline reads; "
             f"phase 2 = {len(rest):,} files / {sum(f['size'] or 0 for f in rest)/1e9:.2f} GB remainder")
    for p in root.rglob("*.part"):                                  # leftovers from an interrupted run
        try:
            p.unlink()
        except OSError:
            pass
    pending1, cached1 = _split_pending(root, core)
    _check_disk(root, pending1, "phase 1")
    failed1, bytes1 = _download_batch(cli, root, pending1, cfg, "phase 1 (pipeline files)")
    report = dict(folder=fid, profile=cfg.download_profile, listed=len(files), selected=len(wanted), phase1_files=len(core), already_complete=cached1,
                  downloaded=len(pending1) - len(failed1), failed=len(failed1), failed_examples=list(failed1.values())[:10], bytes_phase1=bytes1,
                  background_files=len(rest), background="pending" if rest else "none")
    (cfg.cache / "download_report.json").write_text(json.dumps(report, indent=2))
    if failed1:
        LOG.warning(f"[gdrive] {len(failed1)} pipeline files could not be downloaded; first errors: {list(failed1.values())[:3]}")
    else:
        LOG.info(f"[gdrive] phase 1 complete: {report['downloaded']:,} new files, {cached1:,} already present -> pipeline can start")
    if rest:
        pending2, cached2 = _split_pending(root, rest)
        report["background_already_complete"] = cached2
        if pending2:
            try:
                _check_disk(root, pending2, "phase 2")
            except DriveError as e:
                LOG.warning(f"[gdrive] phase 2 skipped: {e}")
                report["background"] = "skipped (disk)"
                return report

            def bg():
                try:
                    f2, b2 = _download_batch(cli, root, pending2, cfg, "phase 2 (remainder, background)")
                    _BG["result"] = dict(downloaded=len(pending2) - len(f2), failed=len(f2), failed_examples=list(f2.values())[:10], bytes=b2)
                except Exception as e:  # noqa: BLE001
                    _BG["error"] = cli._scrub(repr(e))
            _BG.update(thread=threading.Thread(target=bg, daemon=True, name="gdrive-phase2"), result=None, error=None)
            _BG["thread"].start()
            LOG.info(f"[gdrive] phase 2 running in the background ({len(pending2):,} files); the pipeline stages start now")
        else:
            report["background"] = "complete"
    return report


def wait_background(cfg, report=None):
    """Join the phase-2 download (call before finishing the run). Merges its outcome into the report and returns it."""
    th = _BG.get("thread")
    if th is not None and th.is_alive():
        LOG.info("[gdrive] waiting for the background download of the remaining files ...")
        th.join()
    res = _BG.get("result")
    if report is not None:
        if _BG.get("error"):
            report["background"] = f"error: {_BG['error']}"
        elif res is not None:
            report["background"] = "complete" if not res["failed"] else f"{res['failed']} failed"
            report["background_downloaded"] = res["downloaded"]; report["background_failed"] = res["failed"]; report["background_failed_examples"] = res["failed_examples"]
        (cfg.cache / "download_report.json").write_text(json.dumps(report, indent=2))
    _BG["thread"] = None
    return report
