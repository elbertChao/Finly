import argparse
import hashlib
import json
import re
from datetime import date
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
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{primary_doc}"
SEC_SECTION_PATTERNS = {
    "10-K": [
        (r"Item\s+7\.\s+Management['’]s\s+Discussion\s+and\s+Analysis", r"Item\s+7A\."),
        (r"Item\s+1\.\s+Business", r"Item\s+1A\."),
    ],
    "10-Q": [
        (r"Item\s+2\.\s+Management['’]s\s+Discussion\s+and\s+Analysis", r"Item\s+3\."),
        (r"Item\s+1\.\s+Financial\s+Statements", r"Item\s+2\."),
    ],
}
SEC_FINANCE_KEYWORDS = [
    "revenue",
    "sales",
    "growth",
    "margin",
    "gross margin",
    "operating income",
    "operating expenses",
    "cash",
    "liquidity",
    "capital",
    "segments",
    "services",
    "cloud",
    "advertising",
    "guidance",
    "demand",
    "risk",
    "tariff",
    "competition",
    "profit",
    "income taxes",
]


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


def fetch_json(url: str, timeout: int = 20, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="ignore"))


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)</?(div|p|tr|td|th|li|section|article|h[1-6]|br)\b[^>]*>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    lines = [collapse_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def clean_sec_text(text: str) -> str:
    text = re.sub(r"https?://fasb\.org/[^\s]+", " ", text)
    text = re.sub(r"\b(?:us-gaap|xbrli|iso4217|dei|aapl):[A-Za-z0-9:._-]+\b", " ", text)
    text = re.sub(r"\b\d{10}\b", " ", text)
    text = re.sub(r"\bP\d+[A-Z]?\b", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = collapse_whitespace(raw_line)
        if not line:
            continue
        if re.fullmatch(r"[\d\s\-\.,()/%:]{8,}", line):
            continue
        cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines)


def extract_sec_section(text: str, form: str) -> str:
    patterns = SEC_SECTION_PATTERNS.get(form, [])
    for start_pattern, end_pattern in patterns:
        start_matches = list(re.finditer(start_pattern, text, flags=re.IGNORECASE))
        if not start_matches:
            continue

        best_candidate = ""
        for start_match in start_matches:
            trailing_text = text[start_match.end():]
            end_match = re.search(end_pattern, trailing_text, flags=re.IGNORECASE)
            if end_match:
                candidate = text[start_match.start(): start_match.end() + end_match.start()]
            else:
                candidate = text[start_match.start():]

            if len(candidate) > len(best_candidate):
                best_candidate = candidate

        if best_candidate:
            return best_candidate

    return text


def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def score_sec_paragraph(paragraph: str) -> int:
    lowered = paragraph.lower()
    return sum(lowered.count(keyword) for keyword in SEC_FINANCE_KEYWORDS)


def build_ranked_chunks(
    text: str,
    chunk_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
) -> list[dict]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current_paragraphs = []
    current_words = 0
    start_index = 0

    for index, paragraph in enumerate(paragraphs):
        paragraph_words = len(paragraph.split())
        if paragraph_words > chunk_words:
            paragraph = limit_words(paragraph, chunk_words)
            paragraph_words = len(paragraph.split())

        if current_paragraphs and current_words + paragraph_words > chunk_words:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                {
                    "text": chunk_text,
                    "score": sum(score_sec_paragraph(p) for p in current_paragraphs),
                    "start_paragraph": start_index,
                    "end_paragraph": index - 1,
                }
            )

            overlap_paragraphs = []
            overlap_words = 0
            for existing in reversed(current_paragraphs):
                existing_words = len(existing.split())
                if overlap_words + existing_words > chunk_overlap_words:
                    break
                overlap_paragraphs.insert(0, existing)
                overlap_words += existing_words

            current_paragraphs = overlap_paragraphs[:]
            current_words = sum(len(item.split()) for item in current_paragraphs)
            start_index = index - len(current_paragraphs)

        if not current_paragraphs:
            start_index = index

        current_paragraphs.append(paragraph)
        current_words += paragraph_words

    if current_paragraphs:
        chunks.append(
            {
                "text": "\n\n".join(current_paragraphs),
                "score": sum(score_sec_paragraph(p) for p in current_paragraphs),
                "start_paragraph": start_index,
                "end_paragraph": len(paragraphs) - 1,
            }
        )

    chunks.sort(key=lambda item: (item["score"], len(item["text"].split())), reverse=True)
    return chunks[:max_chunks]


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


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


def normalize_cik(cik: str) -> str:
    digits = re.sub(r"\D", "", cik)
    if not digits:
        raise ValueError(f"Invalid CIK value: {cik}")
    return digits.zfill(10)


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


def parse_sec_company_specs(values: list[str]) -> list[dict]:
    companies = []
    for raw in values:
        parts = [part.strip() for part in raw.split(":", maxsplit=1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "Invalid --sec-company value. Use the format TICKER:CIK, for example AAPL:320193."
            )
        companies.append({"ticker": parts[0].upper(), "cik": normalize_cik(parts[1])})
    return companies


def build_sec_filing_url(cik: str, accession_number: str, primary_doc: str) -> str:
    return SEC_ARCHIVES_URL.format(
        cik_nolead=str(int(cik)),
        accession_nodash=accession_number.replace("-", ""),
        primary_doc=primary_doc,
    )


def fetch_sec_samples(
    companies: list[dict],
    user_agent: str,
    forms: set[str],
    filings_per_company: int,
    max_words: int,
    min_filing_date: str | None,
    min_extracted_words: int,
    chunk_words: int,
    chunk_overlap_words: int,
    max_chunks_per_filing: int,
) -> list[dict]:
    samples = []
    min_date = parse_iso_date(min_filing_date) if min_filing_date else None

    for company in companies:
        ticker = company["ticker"]
        cik = company["cik"]
        submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        print(f"Fetching SEC submissions for {ticker} (CIK {cik})")
        payload = fetch_json(submissions_url, user_agent=user_agent)

        company_name = payload.get("name", ticker)
        recent = payload.get("filings", {}).get("recent", {})
        recent_forms = recent.get("form", [])
        recent_accessions = recent.get("accessionNumber", [])
        recent_primary_docs = recent.get("primaryDocument", [])
        recent_dates = recent.get("filingDate", [])

        matched = 0
        candidate_rows = list(
            zip(
                recent_forms,
                recent_accessions,
                recent_primary_docs,
                recent_dates,
            )
        )
        candidate_rows.sort(key=lambda row: row[3], reverse=True)

        for form, accession, primary_doc, filing_date in candidate_rows:
            if min_date and parse_iso_date(filing_date) < min_date:
                continue
            if form not in forms:
                continue

            filing_url = build_sec_filing_url(cik, accession, primary_doc)
            html = fetch_url(filing_url)
            title = f"{company_name} {form} filed {filing_date}"
            filing_text = extract_article_text(html, min_paragraph_chars=80)
            if len(filing_text) < 500:
                filing_text = strip_html(html)

            filing_text = clean_sec_text(filing_text)
            filing_text = extract_sec_section(filing_text, form=form)
            filing_text = limit_words(filing_text, max_words=max_words)
            if len(filing_text.split()) < min_extracted_words:
                print(
                    f"Skipping {ticker} {form} filed {filing_date}: extracted section too short "
                    f"({len(filing_text.split())} words)"
                )
                continue

            ranked_chunks = build_ranked_chunks(
                text=filing_text,
                chunk_words=chunk_words,
                chunk_overlap_words=chunk_overlap_words,
                max_chunks=max_chunks_per_filing,
            )
            if not ranked_chunks:
                continue

            for chunk_index, chunk in enumerate(ranked_chunks):
                if len(chunk["text"].split()) < min_extracted_words:
                    continue

                sample = build_sample(
                    raw_text=chunk["text"],
                    source=filing_url,
                    title=title,
                    extra_metadata={
                        "source_type": "sec_filing",
                        "ticker": ticker,
                        "company_name": company_name,
                        "cik": cik,
                        "form": form,
                        "filing_date": filing_date,
                        "accession_number": accession,
                        "truncated_to_words": max_words,
                        "chunk_index": chunk_index,
                        "chunk_score": chunk["score"],
                        "chunk_start_paragraph": chunk["start_paragraph"],
                        "chunk_end_paragraph": chunk["end_paragraph"],
                    },
                )
                if sample:
                    samples.append(sample)

            matched += 1

            if matched >= filings_per_company:
                break

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
    parser.add_argument(
        "--sec-company",
        action="append",
        default=[],
        help="SEC company spec in the format TICKER:CIK, for example AAPL:320193. Repeat for multiple companies.",
    )
    parser.add_argument(
        "--sec-user-agent",
        default="",
        help="Descriptive User-Agent for SEC requests, ideally including contact information.",
    )
    parser.add_argument(
        "--sec-form",
        action="append",
        default=[],
        help="SEC filing form to include. Repeat for multiple forms.",
    )
    parser.add_argument(
        "--sec-filings-per-company",
        type=int,
        default=2,
        help="Maximum number of recent SEC filings to ingest per company.",
    )
    parser.add_argument(
        "--sec-max-words",
        type=int,
        default=1800,
        help="Maximum number of words to keep from each SEC filing after narrative section extraction.",
    )
    parser.add_argument(
        "--sec-min-filing-date",
        default="",
        help="Optional minimum SEC filing date in YYYY-MM-DD format to avoid stale filings.",
    )
    parser.add_argument(
        "--sec-min-extracted-words",
        type=int,
        default=250,
        help="Minimum acceptable extracted SEC section length before a filing is skipped.",
    )
    parser.add_argument(
        "--sec-chunk-words",
        type=int,
        default=900,
        help="Target words per SEC chunk after section extraction.",
    )
    parser.add_argument(
        "--sec-chunk-overlap-words",
        type=int,
        default=120,
        help="Approximate word overlap between neighboring SEC chunks.",
    )
    parser.add_argument(
        "--sec-max-chunks-per-filing",
        type=int,
        default=2,
        help="Maximum number of ranked SEC chunks to keep from each filing.",
    )

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

    if args.sec_company:
        if not args.sec_user_agent.strip():
            raise ValueError(
                "--sec-user-agent is required when using SEC ingestion so requests identify the project responsibly."
            )
        companies = parse_sec_company_specs(args.sec_company)
        sec_forms = set(args.sec_form) if args.sec_form else {"10-K", "10-Q"}
        samples.extend(
            fetch_sec_samples(
                companies=companies,
                user_agent=args.sec_user_agent.strip(),
                forms=sec_forms,
                filings_per_company=args.sec_filings_per_company,
                max_words=args.sec_max_words,
                min_filing_date=args.sec_min_filing_date.strip() or None,
                min_extracted_words=args.sec_min_extracted_words,
                chunk_words=args.sec_chunk_words,
                chunk_overlap_words=args.sec_chunk_overlap_words,
                max_chunks_per_filing=args.sec_max_chunks_per_filing,
            )
        )

    samples = dedupe_samples(samples)
    if not samples:
        print("No source material found. Provide local .txt files, RSS feeds, or article URL files and rerun.")
        return

    entries = build_jsonl_entries(samples, args.instruction)
    write_jsonl(entries, args.output_jsonl)


if __name__ == "__main__":
    main()
