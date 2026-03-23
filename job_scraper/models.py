"""Pydantic models for the job scraper."""
from pydantic import BaseModel
from typing import Optional, List


class JobListing(BaseModel):
    title: str
    location: Optional[str] = None
    employment_type: Optional[str] = None  # Full-time, Part-time, Remote, Hybrid, Contract
    department: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    salary: Optional[str] = None
    apply_url: Optional[str] = None
    source_url: Optional[str] = None  # Page where this job was found


class UserProfile(BaseModel):
    name: str
    desired_roles: List[str]
    skills: List[str]
    experience_years: Optional[int] = None
    education: Optional[str] = None
    languages: Optional[List[str]] = None
    preferred_locations: Optional[List[str]] = None
    employment_type_preference: Optional[List[str]] = None  # e.g. ["remote", "hybrid"]
    additional_info: Optional[str] = None


class MatchResult(BaseModel):
    score: int           # 1–10, where 10 = perfect match
    is_match: bool       # True if score >= threshold
    reason: str          # Brief explanation from LLM


class JobListings(BaseModel):
    """Wrapper so LLMExtractionStrategy returns a typed list of jobs."""
    jobs: List[JobListing]


class Company(BaseModel):
    name: str
    career_url: str
