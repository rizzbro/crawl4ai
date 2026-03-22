"""LLM-based job-profile matching. Scores each job 1-10 against the user profile."""
import json
from typing import Optional

from crawl4ai import LLMConfig

from models import JobListing, UserProfile, MatchResult


MATCH_PROMPT_TEMPLATE = """
You are an expert job application advisor. Evaluate how well the following job listing matches the candidate profile.

## Candidate Profile
{profile_text}

## Job Listing
{job_text}

## Task
Score the match from 1 to 10:
- 10: Perfect match — all key skills, role, and preferences align
- 7-9: Good match — most important criteria met, minor gaps
- 4-6: Partial match — some overlap but notable gaps
- 1-3: Poor match — few or no criteria align

Respond ONLY with a valid JSON object in this exact format:
{{
  "score": <integer 1-10>,
  "is_match": <true if score >= {threshold}, false otherwise>,
  "reason": "<2-3 sentence explanation of the score>"
}}
"""


def _profile_to_text(profile: UserProfile) -> str:
    """Format a UserProfile into readable text for the LLM prompt."""
    lines = [f"Name: {profile.name}"]

    if profile.desired_roles:
        lines.append(f"Desired Roles: {', '.join(profile.desired_roles)}")
    if profile.skills:
        lines.append(f"Skills: {', '.join(profile.skills)}")
    if profile.experience_years is not None:
        lines.append(f"Years of Experience: {profile.experience_years}")
    if profile.education:
        lines.append(f"Education: {profile.education}")
    if profile.languages:
        lines.append(f"Languages: {', '.join(profile.languages)}")
    if profile.preferred_locations:
        lines.append(f"Preferred Locations: {', '.join(profile.preferred_locations)}")
    if profile.employment_type_preference:
        lines.append(f"Employment Type Preference: {', '.join(profile.employment_type_preference)}")
    if profile.additional_info:
        lines.append(f"Additional Info: {profile.additional_info}")

    return "\n".join(lines)


def _job_to_text(job: JobListing) -> str:
    """Format a JobListing into readable text for the LLM prompt."""
    lines = [f"Title: {job.title}"]

    if job.location:
        lines.append(f"Location: {job.location}")
    if job.employment_type:
        lines.append(f"Employment Type: {job.employment_type}")
    if job.department:
        lines.append(f"Department: {job.department}")
    if job.salary:
        lines.append(f"Salary: {job.salary}")
    if job.description:
        lines.append(f"\nDescription:\n{job.description[:1500]}")  # Limit to avoid token overflow
    if job.requirements:
        lines.append(f"\nRequirements:\n{job.requirements[:1000]}")

    return "\n".join(lines)


def match_job_to_profile(
    job: JobListing,
    profile: UserProfile,
    llm_config: LLMConfig,
    threshold: int = 7,
) -> MatchResult:
    """
    Use an LLM to score how well a job matches a user profile.

    Args:
        job: The job listing to evaluate.
        profile: The candidate's profile.
        llm_config: LLM configuration (provider + token).
        threshold: Minimum score to consider a match (default: 7).

    Returns:
        MatchResult with score, is_match flag, and reason.
    """
    from litellm import completion

    profile_text = _profile_to_text(profile)
    job_text = _job_to_text(job)

    prompt = MATCH_PROMPT_TEMPLATE.format(
        profile_text=profile_text,
        job_text=job_text,
        threshold=threshold,
    )

    try:
        response = completion(
            model=llm_config.provider,
            messages=[{"role": "user", "content": prompt}],
            api_key=llm_config.api_token,
            temperature=0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle potential markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        score = int(data.get("score", 0))
        is_match = bool(data.get("is_match", score >= threshold))
        reason = str(data.get("reason", "No reason provided."))

        return MatchResult(score=score, is_match=is_match, reason=reason)

    except Exception as e:
        # On error, return a conservative non-match result
        return MatchResult(
            score=0,
            is_match=False,
            reason=f"Matching failed due to an error: {e}",
        )
