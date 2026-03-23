"""Core scraping logic: deep-crawl a company's career page and extract job listings."""
import json
import re
from urllib.parse import urlparse
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    FilterChain,
    DomainFilter,
    ContentTypeFilter,
    URLPatternFilter,
    KeywordRelevanceScorer,
)
from crawl4ai import LLMConfig

from models import Company, JobListing, JobListings


EXTRACTION_INSTRUCTION = """
You are a job listing extractor. Analyze the page content and extract ALL job openings/positions you find.

For each job listing found, extract:
- title: The exact job title/position name
- location: City, country, or "Remote" / "Hybrid" (in the language given, translate if clear)
- employment_type: e.g. "Full-time", "Part-time", "Remote", "Hybrid", "Contract", "Internship"
- department: Team or department name (e.g. "Engineering", "Marketing", "Sales")
- description: Full job description or summary (include key responsibilities)
- requirements: Required qualifications, skills, and experience
- salary: Salary range or compensation info, if mentioned
- apply_url: The direct URL to apply or view more details about this specific job

IMPORTANT:
- Return an EMPTY LIST [] if this page contains no job listings
- Do NOT invent or hallucinate job listings
- Include every job listing visible on the page
- If a field is not available, set it to null
"""


def _extract_domain(url: str) -> str:
    """Extract the domain (including subdomain) from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower()



def _parse_jobs_from_result(result, debug: bool = False) -> List[JobListing]:
    """Parse extracted content from a single crawl result into JobListing objects."""
    if not result.success:
        if debug:
            print(f"    [SKIP] {result.url} — crawl failed")
        return []
    if not result.extracted_content:
        if debug:
            print(f"    [EMPTY] {result.url} — no extracted_content")
        return []

    if debug:
        preview = result.extracted_content[:300].replace("\n", " ")
        print(f"    [RAW] {result.url}\n          {preview}...")

    try:
        data = json.loads(result.extracted_content)
    except json.JSONDecodeError as e:
        if debug:
            print(f"    [JSON ERROR] {result.url}: {e}")
        return []

    # Handle both wrapped {"jobs": [...]} and bare list/dict formats
    if isinstance(data, dict) and "jobs" in data:
        items = data["jobs"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    jobs = []
    valid_fields = set(JobListing.model_fields.keys())

    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "").strip()
        if not title or title.lower() in ("null", "none", "n/a"):
            continue
        # Only pass valid fields to avoid pydantic errors
        filtered = {k: v for k, v in item.items() if k in valid_fields}
        try:
            job = JobListing(**filtered)
            job.source_url = result.url
            jobs.append(job)
        except Exception:
            continue

    return jobs


def _deduplicate_jobs(jobs: List[JobListing]) -> List[JobListing]:
    """Remove duplicate job listings (same title + location)."""
    seen = set()
    unique = []
    for job in jobs:
        key = (
            re.sub(r"\s+", " ", job.title.lower().strip()),
            re.sub(r"\s+", " ", (job.location or "").lower().strip()),
        )
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


async def crawl_company(
    crawler: AsyncWebCrawler,
    company: Company,
    llm_config: LLMConfig,
    max_depth: int = 2,
    verbose: bool = False,
    debug_urls: bool = False,
) -> List[JobListing]:
    """
    Deep-crawl a company's career page and return all extracted job listings.

    Args:
        crawler: Active AsyncWebCrawler instance.
        company: Company with name and career_url.
        llm_config: LLM configuration for extraction.
        max_depth: BFS crawl depth (2 = overview + detail pages).
        verbose: Print crawl progress.

    Returns:
        List of deduplicated JobListing objects.
    """
    domain = _extract_domain(company.career_url)

    block_extensions = URLPatternFilter(
        patterns=[
            r".*\.(pdf|jpg|jpeg|png|gif|svg|css|js|woff|woff2|xml|json)(\?.*)?$",
        ],
        reverse=True,
    )

    filter_chain = FilterChain([
        DomainFilter(allowed_domains=[domain]),
        ContentTypeFilter(allowed_types=["text/html"]),
        block_extensions,
    ])

    # Job-keyword scorer to prefer job-related pages
    url_scorer = KeywordRelevanceScorer(
        keywords=["job", "stelle", "career", "position", "opening", "role", "vacancy",
                  "bewerbung", "jobs", "karriere"],
        weight=0.8,
    )

    extraction_strategy = None if debug_urls else LLMExtractionStrategy(
        llm_config=llm_config,
        schema=JobListings.model_json_schema(),
        extraction_type="schema",
        instruction=EXTRACTION_INSTRUCTION,
        apply_chunking=False,
        input_format="markdown",
        extra_args={"temperature": 0, "max_tokens": 8000},
        verbose=verbose,
    )

    content_filter = BM25ContentFilter(
        user_query="job position opening career requirements qualifications stelle bewerbung",
        bm25_threshold=0.2,
    )

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            filter_chain=filter_chain,
            url_scorer=url_scorer,
            include_external=False,
        ),
        extraction_strategy=extraction_strategy,
        markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter) if not debug_urls else None,
        cache_mode=CacheMode.ENABLED,
        verbose=verbose,
        page_timeout=30000,
    )

    if verbose:
        print(f"  Crawling: {company.career_url} (depth={max_depth})")

    results = await crawler.arun(url=company.career_url, config=config)

    # arun with deep crawl returns a list
    if not isinstance(results, list):
        results = [results]

    print(f"  → Deep crawl returned {len(results)} page(s)")
    if debug_urls:
        print(f"\n  === Gecrawlte URLs ({len(results)} Seiten) ===")
    for r in results:
        has_content = bool(r.extracted_content) if not debug_urls else None
        if debug_urls:
            status = "OK " if r.success else "ERR"
            print(f"  [{status}] {r.url}")
        else:
            print(f"  [{'content' if has_content else 'empty  '}] {r.url}")
    if debug_urls:
        print()

    all_jobs: List[JobListing] = []
    for result in results:
        jobs = _parse_jobs_from_result(result, debug=verbose)
        if jobs:
            print(f"    [+] {len(jobs)} job(s) on {result.url}")
        all_jobs.extend(jobs)

    unique_jobs = _deduplicate_jobs(all_jobs)

    if verbose:
        print(f"  Total unique jobs found: {len(unique_jobs)}")

    return unique_jobs
