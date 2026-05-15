import os
from dotenv import load_dotenv

load_dotenv()

# Key is validated at runtime when the Anthropic client is constructed.
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-5")

BASE_RESUME_DIR = "output/baseline"
JD_DIR = "data/job_descriptions"
OUTPUT_DIR = "output"
PARSED_RESUME_CACHE = "data/base_resume/parsed_resume.json"
