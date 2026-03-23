#!/usr/bin/env python3
"""
Job Scraper — powered by crawl4ai
==================================
Crawls company career pages, extracts job listings, and saves matches to disk.

Usage examples:
  python main.py --provider anthropic
  python main.py --provider ollama --profile data/profile.json --companies data/companies.json
  python main.py --provider openai --threshold 8 --max-depth 3 --verbose
  python main.py --list-providers
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig

from llm_config import get_llm_config, list_providers
from models import Company, UserProfile
from scraper import crawl_company
from matcher import match_job_to_profile
from output_manager import save_job, save_summary


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape company career pages and find matching jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider", "-p",
        default="anthropic",
        help="LLM provider to use for extraction and matching (default: anthropic). "
             "Use --list-providers to see all options.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model string, e.g. 'ollama/llama3.1' or 'anthropic/claude-3-haiku-20240307'.",
    )
    parser.add_argument(
        "--profile", "-f",
        default="data/profile.json",
        help="Path to the user profile JSON file (default: data/profile.json).",
    )
    parser.add_argument(
        "--companies", "-c",
        default="data/companies.json",
        help="Path to the companies JSON file (default: data/companies.json).",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for saved job files (default: output/).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=7,
        help="Minimum match score (1–10) to save a job (default: 7).",
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=2,
        help="BFS crawl depth for discovering job pages (default: 2).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress output.",
    )
    parser.add_argument(
        "--debug-urls",
        action="store_true",
        help="Print all crawled URLs without running LLM extraction (fast debug mode).",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all available LLM providers and exit.",
    )
    return parser.parse_args()


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_profile(path: str) -> UserProfile:
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"[ERROR] Profile file not found: {path}\n"
            f"Copy data/profile_example.json to {path} and edit it."
        )
    try:
        return UserProfile(**json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        sys.exit(f"[ERROR] Could not parse profile file '{path}': {e}")


def load_companies(path: str) -> list[Company]:
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"[ERROR] Companies file not found: {path}\n"
            f"Copy data/companies_example.json to {path} and edit it."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        companies = [Company(**item) for item in data]
        if not companies:
            sys.exit("[ERROR] Companies file is empty.")
        return companies
    except Exception as e:
        sys.exit(f"[ERROR] Could not parse companies file '{path}': {e}")


# ── Main async logic ──────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    # Build LLM config
    try:
        llm_config = get_llm_config(args.provider, model_override=args.model)
    except (ValueError, EnvironmentError) as e:
        sys.exit(f"[ERROR] {e}\n\nAvailable providers:\n{list_providers()}")

    profile = load_profile(args.profile)
    companies = load_companies(args.companies)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Job Scraper — Profile: {profile.name}")
    print(f"  LLM Provider : {llm_config.provider}")
    print(f"  Companies    : {len(companies)}")
    print(f"  Match threshold: {args.threshold}/10")
    print(f"  Output dir   : {output_dir.resolve()}")
    print(f"{'='*60}\n")

    all_results = []  # [(Company, [(JobListing, MatchResult)])]

    browser_config = BrowserConfig(headless=True, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, company in enumerate(companies, 1):
            print(f"[{i}/{len(companies)}] Processing: {company.name}")
            print(f"  Career URL: {company.career_url}")

            try:
                jobs = await crawl_company(
                    crawler=crawler,
                    company=company,
                    llm_config=llm_config,
                    max_depth=args.max_depth,
                    verbose=args.verbose,
                    debug_urls=args.debug_urls,
                )
            except Exception as e:
                print(f"  [WARNING] Crawling failed for {company.name}: {e}")
                all_results.append((company, []))
                continue

            if not jobs:
                print(f"  No job listings extracted.")
                all_results.append((company, []))
                continue

            print(f"  Extracted {len(jobs)} job(s). Now matching against profile...")

            matched_jobs = []
            for job in jobs:
                match = match_job_to_profile(
                    job=job,
                    profile=profile,
                    llm_config=llm_config,
                    threshold=args.threshold,
                )

                # Always show all scores so we can diagnose threshold issues
                marker = "✓" if match.is_match else "✗"
                print(f"  {marker} [{match.score}/10] {job.title}")
                if args.verbose:
                    print(f"      {match.reason[:120]}")

                if match.is_match:
                    file_path = save_job(
                        job=job,
                        match=match,
                        company=company,
                        profile=profile,
                        output_dir=output_dir,
                    )
                    print(f"  ✓ [{match.score}/10] {job.title}")
                    if args.verbose:
                        print(f"    Saved: {file_path}")
                    matched_jobs.append((job, match))
                else:
                    if args.verbose:
                        print(f"  ✗ [{match.score}/10] {job.title} (below threshold)")

            if not matched_jobs:
                print(f"  No matching jobs found (threshold: {args.threshold}/10).")

            all_results.append((company, matched_jobs))
            print()

    # Write summary
    summary_path = save_summary(all_results, profile, output_dir)

    total_matches = sum(len(jobs) for _, jobs in all_results)
    print(f"\n{'='*60}")
    print(f"  Done! {total_matches} matching job(s) saved.")
    print(f"  Summary: {summary_path}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.list_providers:
        print("Available LLM providers:\n")
        print(list_providers())
        sys.exit(0)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
