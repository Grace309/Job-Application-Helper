"""Use Claude to produce a job-specific customized resume from a base resume."""

import json
import copy
from anthropic import Anthropic

from config import CLAUDE_MODEL


def merge_resumes(
    resume_a: dict,
    resume_b: dict,
    client: Anthropic,
) -> dict:
    """
    Use Claude to intelligently merge two parsed resumes into one best-of-both version.
    Keeps all real content from both; picks the strongest bullets and framing.
    """
    prompt = f"""You are an expert resume writer and career coach.
You have been given TWO versions of the same person's resume, each formatted differently and emphasizing different aspects.

Your task is to merge them into ONE comprehensive, high-quality resume that:
1. **Keeps ALL unique content** — if an experience bullet, project, skill, or achievement appears in either version, include it (unless it is a strictly weaker duplicate).
2. **Picks the STRONGER wording** — when both versions describe the same item, choose the more impactful, specific, and quantified phrasing.
3. **Preserves accuracy** — do NOT invent new facts, metrics, or experiences not present in either version.
4. **Uses the contact info** from Resume A (treat it as authoritative for name, email, phone, linkedin, github, location).
5. **Bullets** — keep each bullet ≤ 2 lines (~25 words). Use strong action verbs.
6. **Projects** — include all unique projects from both versions.
7. **Skills** — union of all skills from both versions, deduplicated.

═══════════════════════════ RESUME A ═══════════════════════════
{json.dumps(resume_a, indent=2, ensure_ascii=False)}

═══════════════════════════ RESUME B ═══════════════════════════
{json.dumps(resume_b, indent=2, ensure_ascii=False)}

Return ONLY the merged resume as valid JSON with the EXACT same schema as the inputs.
No markdown fences, no commentary — only raw JSON.
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```", 2)[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.rsplit("```", 1)[0].strip()

    try:
        merged = json.loads(response_text)
    except json.JSONDecodeError:
        print("  WARNING: Merge returned invalid JSON – falling back to Resume A.")
        merged = copy.deepcopy(resume_a)

    return merged


def customize_resume(
    base_resume: dict,
    job_description: str,
    company: str,
    position: str,
    client: Anthropic,
) -> dict:
    """Return a deep-customized copy of *base_resume* tailored for the given JD."""

    prompt = f"""You are an expert resume writer and career coach.
Your task is to customize the provided resume for a specific job application.

═══════════════════════════ JOB INFORMATION ═══════════════════════════
Company   : {company}
Position  : {position}

Job Description:
\"\"\"
{job_description}
\"\"\"

═══════════════════════════ BASE RESUME (JSON) ═══════════════════════
{json.dumps(base_resume, indent=2, ensure_ascii=False)}

═══════════════════════════ INSTRUCTIONS ════════════════════════════
Produce a CUSTOMIZED version of the resume that:

1. **Summary** – Rewrite to directly address this role, company, and seniority level.
2. **Experience bullets** – Reorder and rewrite bullets to surface the most relevant
   achievements. Use strong action verbs. Quantify where the base resume already
   contains numbers; do NOT invent new metrics.
3. **Skills** – Reorder skill lists so the most relevant items appear first.
   Remove skills clearly unrelated to this role if it helps clarity.
4. **Projects** – Keep max 3 projects; prefer those most relevant to the JD.
5. **Keywords** – Naturally embed important JD keywords throughout.

ABSOLUTE RULES (violations make the resume fraudulent):
- Do NOT add experience, companies, jobs, degrees, or certifications not in the base.
- Do NOT change company names, job titles, institutions, or any dates.
- Do NOT invent metrics or numbers not present in the original.
- You MAY reword, expand, condense, or reorder existing content.
- Keep each bullet ≤ 2 lines (roughly 25 words).

Return ONLY the customized resume as valid JSON with the EXACT same schema as the
base resume above. No markdown fences, no commentary — only raw JSON.
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=6144,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```", 2)[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.rsplit("```", 1)[0].strip()

    try:
        customized = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: return original resume if Claude output is malformed
        print("  WARNING: Claude returned invalid JSON – falling back to base resume.")
        customized = copy.deepcopy(base_resume)

    return customized


def refine_resume(
    current_resume: dict,
    base_resume: dict,
    job_description: str,
    company: str,
    position: str,
    score: dict,
    client: Anthropic,
) -> dict:
    """
    Given a scored resume, use the feedback to produce an improved version.
    Never adds content not present in base_resume.
    """
    improvements = "\n".join(f"- {i}" for i in score.get("improvements", []))
    strengths = "\n".join(f"- {s}" for s in score.get("strengths", []))

    prompt = f"""You are an expert resume writer. A resume was evaluated against a job description and received a score of {score.get('total', '?')}/{score.get('out_of', 50)} (Grade: {score.get('grade', '?')}).

Your task is to produce an IMPROVED version that addresses the feedback below — but ONLY using content already present in the base resume. Do NOT invent new experience, metrics, or skills.

═══════════════════════════ JOB INFORMATION ═══════════════════════════
Company  : {company}
Position : {position}

Job Description:
\"\"\"
{job_description}
\"\"\"

═══════════════════════════ EVALUATION FEEDBACK ══════════════════════
Verdict : {score.get('verdict', '')}

Strengths (preserve and amplify these):
{strengths}

Improvements needed (address these using ONLY existing resume content):
{improvements}

═══════════════════════════ BASE RESUME (source of truth) ════════════
{json.dumps(base_resume, indent=2, ensure_ascii=False)}

═══════════════════════════ CURRENT VERSION (to improve) ═════════════
{json.dumps(current_resume, indent=2, ensure_ascii=False)}

═══════════════════════════ INSTRUCTIONS ════════════════════════════
Produce an improved resume by:
1. Better surfacing skills/experience from the base that the current version underemphasises.
2. Rewriting bullets to use stronger, more specific language aligned with JD keywords.
3. Reordering sections/items so the most relevant content appears first.
4. Tightening wording to be more impactful (action verb + result + context).

ABSOLUTE RULES:
- Do NOT add companies, roles, degrees, projects, or certifications not in the base.
- Do NOT invent or alter any numbers/metrics.
- You MAY reword, reorder, and restructure freely.
- Keep each bullet ≤ 2 lines (~25 words).

Return ONLY valid JSON with the same schema. No markdown, no commentary.
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=6144,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```", 2)[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.rsplit("```", 1)[0].strip()

    try:
        refined = json.loads(response_text)
    except json.JSONDecodeError:
        print("  WARNING: Refinement returned invalid JSON – keeping current version.")
        refined = copy.deepcopy(current_resume)

    return refined
