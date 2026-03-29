import argparse
import hashlib
import json
import re
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_INSTRUCTION = (
    "Analyze the long-term growth prospects of this company based on the text context. "
    "Explain the key business drivers, risks, and whether the stock appears better suited "
    "for holding, waiting, or deeper research."
)

DEFAULT_USER_AGENT = "Fin-Instruct/0.1 (dataset research bot; educational use)"


def fetch_url(url: str, timeout: int = 20) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/xml;q=0.8,*/*;q=0.7",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return collapse_whitespace(unescape(text))


def extract_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    return collapse_whitespace(unescape(match.group(1)))


def extract_article_text(html: str, min_paragraph_chars: int = 120) -> str:
    paragraphs = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", html)
    cleaned = []

    for paragraph in paragraphs:
        text = strip_html(paragraph)
        if len(text) >= min_paragraph_chars:
            cleaned.append(text)

    if cleaned:
        return "\n\n".join(cleaned)

    return strip_html(html)


def make_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def build_sample(raw_text: str, source: str, title: str = "", extra_metadata: dict | None = None) -> dict | None:
    normalized = collapse_whitespace(raw_text)
    if not normalized:
        return None

    metadata = {
        "source": source,
        "title": title,
        "source_type": extra_metadata.get("source_type", "unknown") if extra_metadata else "unknown",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "id": make_id(f"{source}|{title}|{normalized[:300]}"),
        "source": source,
        "title": title,
        "raw_text": normalized,
        "metadata": metadata,
    }


def scan_text_inputs(input_dir: str) -> list[dict]:
    data = []
    root = Path(input_dir)
    if not root.exists():
        return data

    for path in sorted(root.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        sample = build_sample(
            raw_text=raw,
            source=str(path),
            title=path.stem,
            extra_metadata={"source_type": "local_text"},
        )
        if sample:
            data.append(sample)
    return data


def parse_rss_feed(feed_url: str, item_limit: int) -> list[dict]:
    xml_text = fetch_url(feed_url)
    root = ET.fromstring(xml_text)
    samples = []

    for item in root.findall(".//item")[:item_limit]:
        title = collapse_whitespace(item.findtext("title", default=""))
        link = collapse_whitespace(item.findtext("link", default=""))
        description = collapse_whitespace(item.findtext("description", default=""))

        sample = build_sample(
            raw_text=f"{title}\n\n{description}",
            source=link or feed_url,
            title=title,
            extra_metadata={
                "source_type": "rss",
                "feed_url": feed_url,
            },
        )
        if sample:
            samples.append(sample)

    return samples


def read_urls(url_file: str) -> list[str]:
    urls = []
    for line in Path(url_file).read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            urls.append(candidate)
    return urls


def fetch_article_samples(urls: Iterable[str]) -> list[dict]:
    samples = []
    for url in urls:
        html = fetch_url(url)
        title = extract_title(html)
        article_text = extract_article_text(html)
        domain = urlparse(url).netloc

        sample = build_sample(
            raw_text=article_text,
            source=url,
            title=title,
            extra_metadata={
                "source_type": "article_url",
                "domain": domain,
            },
        )
        if sample:
            samples.append(sample)
    return samples


def dedupe_samples(samples: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for sample in samples:
        key = (sample["source"], sample["raw_text"][:500])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def build_jsonl_entries(samples: list[dict], instruction: str) -> list[dict]:
    entries = []
    for sample in samples:
        entries.append(
            {
                "instruction": instruction,
                "context": sample["raw_text"],
                "response": "",
                "metadata": {
                    "id": sample["id"],
                    "source": sample["source"],
                    "title": sample.get("title", ""),
                    **sample.get("metadata", {}),
                },
            }
        )
    return entries


def write_jsonl(entries: list[dict], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in entries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Curate financial dataset JSONL from local files and online sources")
    parser.add_argument("--input-dir", default="inputs", help="Directory containing local .txt files")
    parser.add_argument("--output-jsonl", default="data/curated_dataset.jsonl", help="Output JSONL file path")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Instruction to use for all examples")
    parser.add_argument("--feed-url", action="append", default=[], help="RSS feed URL to ingest; repeat for multiple feeds")
    parser.add_argument("--feed-item-limit", type=int, default=10, help="Max items to ingest per RSS feed")
    parser.add_argument("--url-file", action="append", default=[], help="Text file containing article URLs to fetch")

    args = parser.parse_args()

    samples = []
    samples.extend(scan_text_inputs(args.input_dir))

    for feed_url in args.feed_url:
        print(f"Fetching RSS feed: {feed_url}")
        samples.extend(parse_rss_feed(feed_url, item_limit=args.feed_item_limit))

    article_urls = []
    for url_file in args.url_file:
        article_urls.extend(read_urls(url_file))

    if article_urls:
        print(f"Fetching {len(article_urls)} article URLs")
        samples.extend(fetch_article_samples(article_urls))

    samples = dedupe_samples(samples)
    if not samples:
        print("No source material found. Provide local .txt files, RSS feeds, or article URL files and rerun.")
        return

    entries = build_jsonl_entries(samples, args.instruction)
    write_jsonl(entries, args.output_jsonl)


if __name__ == "__main__":
    main()
