"""Save matched job listings as Markdown files in a structured folder hierarchy."""
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from models import Company, JobListing, MatchResult, UserProfile


def _sanitize_filename(name: str, max_length: int = 80) -> str:
    """Convert a string into a safe filename."""
    # Replace special chars with hyphens
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", name)
    # Collapse multiple spaces/hyphens
    safe = re.sub(r"[-\s]+", "-", safe).strip("-")
    return safe[:max_length]


def _format_job_markdown(
    job: JobListing,
    match: MatchResult,
    company: Company,
    profile: UserProfile,
) -> str:
    """Render a job listing and its match result as a Markdown document."""

    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# {job.title}",
        f"",
        f"**Company:** {company.name}  ",
        f"**Match Score:** {match.score}/10 {'⭐' * match.score}",
        f"**Scraped:** {scraped_at}",
        f"",
        "---",
        "",
        "## Match Analysis",
        f"**Score:** {match.score}/10",
        f"**Profile:** {profile.name}",
        f"**Assessment:** {match.reason}",
        "",
        "---",
        "",
        "## Job Details",
        "",
    ]

    if job.location:
        sections.append(f"**Location:** {job.location}")
    if job.employment_type:
        sections.append(f"**Employment Type:** {job.employment_type}")
    if job.department:
        sections.append(f"**Department:** {job.department}")
    if job.salary:
        sections.append(f"**Salary:** {job.salary}")
    if job.apply_url:
        sections.append(f"**Apply:** [{job.apply_url}]({job.apply_url})")
    if job.source_url:
        sections.append(f"**Source:** [{job.source_url}]({job.source_url})")

    sections.append("")

    if job.description:
        sections += ["### Description", "", job.description, ""]

    if job.requirements:
        sections += ["### Requirements", "", job.requirements, ""]

    return "\n".join(sections)


def save_job(
    job: JobListing,
    match: MatchResult,
    company: Company,
    profile: UserProfile,
    output_dir: str | Path,
) -> Path:
    """
    Save a single matched job as a Markdown file.

    Structure: <output_dir>/<CompanyName>/<job_title>.md

    Returns the path of the saved file.
    """
    output_dir = Path(output_dir)
    company_dir = output_dir / _sanitize_filename(company.name)
    company_dir.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(f"{match.score:02d}_{job.title}") + ".md"
    file_path = company_dir / filename

    # If a file with the same name exists, append a counter
    counter = 1
    while file_path.exists():
        stem = _sanitize_filename(f"{match.score:02d}_{job.title}_{counter}")
        file_path = company_dir / (stem + ".md")
        counter += 1

    content = _format_job_markdown(job, match, company, profile)
    file_path.write_text(content, encoding="utf-8")

    return file_path


def save_summary(
    results: List[Tuple[Company, List[Tuple[JobListing, MatchResult]]]],
    profile: UserProfile,
    output_dir: str | Path,
) -> Path:
    """
    Write a top-level summary Markdown file listing all matches across companies.

    Returns path to the summary file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Job Match Summary",
        f"",
        f"**Profile:** {profile.name}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        "---",
        "",
    ]

    total_matches = 0

    for company, jobs in results:
        if not jobs:
            continue
        lines.append(f"## {company.name}")
        lines.append(f"Career page: [{company.career_url}]({company.career_url})")
        lines.append("")
        lines.append(f"| Score | Job Title | Location | Employment Type |")
        lines.append(f"|-------|-----------|----------|-----------------|")

        for job, match in sorted(jobs, key=lambda x: x[1].score, reverse=True):
            location = job.location or "—"
            etype = job.employment_type or "—"
            lines.append(f"| {match.score}/10 | {job.title} | {location} | {etype} |")
            total_matches += 1

        lines.append("")

    lines += [
        "---",
        f"",
        f"**Total matching positions found:** {total_matches}",
    ]

    summary_path = output_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
