"""ThumbnailManager — Download, validate, cache thumbnails."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from PIL import Image
from tqdm.auto import tqdm


class ThumbnailManager:
    """Centralized thumbnail management — single source of truth for all thumbnail operations."""

    def __init__(self, config, target_size=(224, 224), min_size=(120, 80)):
        self.thumb_dir = f"{config['data']['raw_dir']}/thumbnails"
        self.template = config["data"]["thumbnail_url_template"]
        self.fallback = config["data"]["thumbnail_fallback_url"]
        self.workers = config["data"]["thumbnail_download_workers"]
        self.rate_limit = config["data"].get("thumbnail_rate_limit", 10)
        self.target_size = target_size
        self.min_size = min_size
        self.lock = threading.Lock()
        self.last_t = 0.0
        os.makedirs(self.thumb_dir, exist_ok=True)

    def _rate_limit_wait(self):
        min_gap = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        with self.lock:
            now = time.time()
            wait = min_gap - (now - self.last_t)
            if wait < 0:
                wait = 0.0
        if wait > 0:
            time.sleep(wait)
        with self.lock:
            self.last_t = time.time()

    def _download_one(self, vid):
        self._rate_limit_wait()
        path = os.path.join(self.thumb_dir, f"{vid}.jpg")
        if os.path.exists(path):
            try:
                img = Image.open(path)
                img.verify()
                if img.size == self.target_size:
                    return vid, "exists"
                img.close()
                os.remove(path)
            except Exception:
                try:
                    os.remove(path)
                except Exception:
                    pass
        try:
            import requests
            r = requests.get(self.template.format(video_id=vid), timeout=5)
            if r.status_code != 200:
                r = requests.get(self.fallback.format(video_id=vid), timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                if img.size[0] < self.min_size[0] or img.size[1] < self.min_size[1]:
                    return vid, False
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                img.save(path, quality=95)
                img = Image.open(path)
                img.verify()
                return vid, True
        except Exception:
            pass
        return vid, False

    def download(self, video_ids, desc="Thumbnails"):
        ok = fail = skip = 0
        failed = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._download_one, v): v for v in video_ids}
            for fut in tqdm(as_completed(futs), total=len(video_ids), desc=desc):
                vid, status = fut.result()
                if status is True:
                    ok += 1
                elif status is False:
                    fail += 1
                    failed.append(vid)
                else:
                    skip += 1
        print(f"[Thumbnails] OK: {ok:,}  |  Failed: {fail:,}  |  Cached: {skip:,}")
        return failed

    def get_valid(self, video_ids):
        valid = []
        missing = []
        for vid in video_ids:
            path = os.path.join(self.thumb_dir, f"{vid}.jpg")
            try:
                img = Image.open(path)
                img.verify()
                valid.append(vid)
            except Exception:
                missing.append(vid)
        return valid, missing

    def load_image(self, video_id):
        path = os.path.join(self.thumb_dir, f"{video_id}.jpg")
        return Image.open(path).convert("RGB")
