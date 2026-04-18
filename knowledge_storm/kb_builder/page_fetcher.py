import os
import re
from typing import Dict

import trafilatura


class PageFetcher:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._url_to_path: Dict[str, str] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def fetch(self, url: str, fallback_snippet: str = "") -> str:
        if url in self._url_to_path:
            with open(self._url_to_path[url], encoding="utf-8") as f:
                return f.read()

        content = self._fetch_url(url)
        if content is None:
            return fallback_snippet

        cache_path = os.path.join(self.cache_dir, self._slugify(url) + ".md")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._url_to_path[url] = cache_path
        return content

    def already_fetched(self, url: str) -> bool:
        return url in self._url_to_path

    def _fetch_url(self, url: str):
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return None
            return trafilatura.extract(
                downloaded,
                output_format="markdown",
                include_links=False,
                include_images=False,
                include_formatting=True,
            )
        except Exception:
            return None

    def _slugify(self, text: str) -> str:
        text = re.sub(r"https?://", "", text)
        text = re.sub(r"[^\w\-]", "-", text)
        return text[:80].strip("-")
