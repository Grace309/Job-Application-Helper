#!/usr/bin/env python3
"""
Job Application Helper — CLI entry point

Usage examples:
  # First run: parse base resume and generate for one JD
  python main.py -j data/job_descriptions/google_swe.txt -c Google -p "Software Engineer"

  # Reuse cached parsed resume (skip API call for parsing)
  python main.py -j data/job_descriptions/google_swe.txt -c Google -p "Software Engineer" --cached

  # Specify a different base resume PDF
  python main.py -r path/to/my_resume.pdf -j data/job_descriptions/amazon.txt -c Amazon -p "SDE II"

  # Batch: process every .txt file in the JD directory
  python main.py --batch -c "" -p ""
"""

from __future__ import annotations

import argparse
import html as _html_module
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import csv
from anthropic import Anthropic

import config
from src.pdf_parser import parse_resume_to_json
from src.resume_customizer import customize_resume, refine_resume, merge_resumes
from src.latex_generator import generate_latex
from src.compiler import compile_latex
from src.scorer import score_resume, print_score_report
from src.cover_letter_generator import build_cover_letter


def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text).strip("_")


TRACKER_PATH = Path(config.OUTPUT_DIR) / "applications.csv"
_HEADERS = [
    "Company", "Position", "Applied Date", "Score", "Grade",
    "Keyword Match", "Experience Relevance", "Impact & Clarity",
    "Skills Alignment", "Overall Fit", "Verdict", "Link", "Output Folder",
]


def _update_tracker(
    company: str,
    position: str,
    score: dict,
    job_dir: Path,
    link: str = "",
) -> None:
    """Append one row to the applications CSV tracker."""
    csv_path = TRACKER_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    scores = score.get("scores", {})
    row = [
        company,
        position,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        f"{score.get('total', '')}/{score.get('out_of', 50)}",
        score.get("grade", ""),
        scores.get("keyword_match", ""),
        scores.get("experience_relevance", ""),
        scores.get("impact_clarity", ""),
        scores.get("skills_alignment", ""),
        scores.get("overall_fit", ""),
        score.get("verdict", ""),
        link,
        str(job_dir.resolve()),
    ]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_HEADERS)
        writer.writerow(row)
    print(f"  Tracker : {csv_path}")


def _fetch_url_text(url: str) -> str:
    """Fetch a URL and return stripped plain text."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            charset = resp.headers.get_content_charset("utf-8")
            raw = resp.read().decode(charset, errors="replace")
    except Exception as exc:
        print(f"  ERROR fetching URL: {exc}")
        sys.exit(1)
    raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = _html_module.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _extract_jd_info(raw_text: str, client: Anthropic) -> tuple[str, str, str]:
    """Use Claude to pull company, position title, and clean JD from raw text."""
    if len(raw_text.strip()) < 200:
        print("  WARNING: page content is very short — the site may be JS-rendered.")
        print("  TIP: paste the JD text directly with --jd, or use a Greenhouse/Lever URL.")
        sys.exit(1)

    prompt = f"""Extract information from this job posting content.

Content:
\"\"\"
{raw_text[:12000]}
\"\"\"

Return ONLY a JSON object with these exact keys:
- "company": the company name
- "position": the job title / position
- "jd_clean": clean job description containing only: brief company intro (1-2 sentences), responsibilities, requirements, and tech stack — remove navigation menus, application forms, cookie notices, legal boilerplate, footer links, and anything unrelated to the role itself

Return ONLY valid JSON, no markdown fences, no extra text."""

    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    # Strip markdown code fences if Claude wrapped the response
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        print("  ERROR: Claude returned empty response. The page content may be too sparse.")
        print("  TIP: paste the JD text directly with --jd, or use a Greenhouse/Lever URL.")
        sys.exit(1)
    data = json.loads(text)
    return data["company"], data["position"], data["jd_clean"]


def _find_base_resume() -> list[Path]:
    """Return all PDFs found in the base resume directory."""
    base_dir = Path(config.BASE_RESUME_DIR)
    pdfs = sorted(base_dir.glob("*.pdf"))
    if not pdfs:
        print(
            f"ERROR: No PDF found in {base_dir}/\n"
            f"  Place your base resume there and try again, or use --resume <path>."
        )
        sys.exit(1)
    return pdfs


def _load_or_parse_resume(resume_path: Path | None, client: Anthropic, use_cache: bool) -> dict:
    cache_path = Path(config.PARSED_RESUME_CACHE)
    merged_cache_path = cache_path.parent / "parsed_resume_merged.json"

    # Use merged cache if it exists and no explicit resume path given
    if merged_cache_path.exists() and (use_cache or resume_path is None):
        print("  Loading cached merged resume …")
        with open(merged_cache_path, encoding="utf-8") as f:
            return json.load(f)

    # Auto-use single cache if it exists and no explicit resume path given
    if cache_path.exists() and (use_cache or resume_path is None):
        print("  Loading cached parsed resume …")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    if resume_path is not None:
        # Single explicit resume path
        print("  Parsing resume with Claude …")
        resume_data = parse_resume_to_json(str(resume_path), client)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(resume_data, f, indent=2, ensure_ascii=False)
        print(f"  Parsed resume cached at {cache_path}")
        return resume_data

    # Auto-detect PDFs in BASE_RESUME_DIR
    pdfs = _find_base_resume()

    if len(pdfs) == 1:
        print(f"  Parsing resume: {pdfs[0].name} …")
        resume_data = parse_resume_to_json(str(pdfs[0]), client)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(resume_data, f, indent=2, ensure_ascii=False)
        print(f"  Parsed resume cached at {cache_path}")
        return resume_data

    # Multiple PDFs — parse all and merge into one best-of-all resume
    print(f"  Found {len(pdfs)} resume PDFs — parsing and merging into best-of-all …")
    parsed = []
    for pdf in pdfs:
        print(f"    Parsing: {pdf.name} …")
        parsed.append(parse_resume_to_json(str(pdf), client))

    merged = parsed[0]
    for other in parsed[1:]:
        print(f"    Merging with Claude …")
        merged = merge_resumes(merged, other, client)

    merged_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_cache_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Merged resume cached at {merged_cache_path}")
    return merged


def _process_one(
    base_resume: dict,
    jd_path: Path,
    company: str,
    position: str,
    output_dir: Path,
    client: Anthropic,
    optimize: bool = True,
    cover: bool = False,
    link: str = "",
) -> None:
    print(f"\n{'─'*60}")
    print(f"  JD      : {jd_path}")
    print(f"  Company : {company or '(unknown)'}")
    print(f"  Position: {position or '(unknown)'}")

    with open(jd_path, encoding="utf-8") as f:
        jd_text = f.read()

    # Derive names from JD filename if not provided
    _company = company or jd_path.stem.split("_")[0]
    _position = position or jd_path.stem

    print("  Customizing resume with Claude …")
    customized = customize_resume(base_resume, jd_text, _company, _position, client)

    # Score then refine once
    print("  Scoring …")
    score = score_resume(customized, jd_text, _company, _position, client)
    print(f"  Score: {score.get('total')}/{score.get('out_of')} ({score.get('grade')}) — refining …")
    customized = refine_resume(customized, base_resume, jd_text, _company, _position, score, client)

    # Per-job output subfolder: output/<Company>_<Position>/
    stem = _safe_filename(f"{_company}_{_position}")
    job_dir = output_dir / stem
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save customized JSON for auditing
    json_out = job_dir / f"{stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(customized, f, indent=2, ensure_ascii=False)

    # Generate LaTeX
    latex_content = generate_latex(customized)
    tex_out = job_dir / f"{stem}.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(latex_content)

    # Compile to PDF
    pdf_path = compile_latex(tex_out, job_dir)
    if pdf_path:
        tex_out.unlink(missing_ok=True)  # keep only PDF
        print(f"  PDF     : {pdf_path}")
    else:
        print(f"  PDF     : (compile manually – see {tex_out})")

    # Cover letter (only if requested)
    if cover:
        build_cover_letter(customized, jd_text, _company, _position, job_dir, stem, client)

    # Final score
    print("  Scoring resume with Claude …")
    score = score_resume(customized, jd_text, _company, _position, client)
    score_json = job_dir / f"{stem}_score.json"
    with open(score_json, "w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    print_score_report(score, _company, _position)

    # Update Excel tracker
    _update_tracker(_company, _position, score, job_dir, link)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Generate a customized LaTeX/PDF resume for a job application.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--resume", "-r", metavar="PDF", help="Path to base resume PDF (auto-detected if omitted and cache exists).")
    parser.add_argument(
        "--url", "-u", metavar="URL",
        help="Job posting URL — fetches and extracts JD automatically.",
    )
    parser.add_argument(
        "--jd", "-j", metavar="FILE",
        help="Path to a job description .txt file.",
    )
    parser.add_argument("--company", "-c", metavar="NAME", default="", help="Company name (auto-extracted if omitted).")
    parser.add_argument("--position", "-p", metavar="TITLE", default="", help="Position title (auto-extracted if omitted).")
    parser.add_argument(
        "--output", "-o", metavar="DIR",
        default=config.OUTPUT_DIR,
        help=f"Output directory (default: {config.OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cached", action="store_true",
        help="Force use of cached parsed resume JSON.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help=f"Process ALL .txt files in {config.JD_DIR}/.",
    )
    parser.add_argument(
        "--cover", action="store_true",
        help="Also generate a cover letter PDF (off by default).",
    )
    args = parser.parse_args()

    if not args.batch and not args.jd and not args.url:
        parser.error("Provide --url <url>, --jd <file>, or --batch.")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    output_dir = Path(args.output)

    # Resolve base resume (None = auto-detect / use cache)
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and not resume_path.exists():
        print(f"ERROR: Resume not found: {resume_path}")
        sys.exit(1)

    # Parse / load resume
    base_resume = _load_or_parse_resume(resume_path, client, args.cached)

    # ── URL mode ──────────────────────────────────────────────
    if args.url:
        print(f"  Fetching JD from URL …")
        raw = _fetch_url_text(args.url)
        print(f"  Extracting company / position / JD with Claude …")
        company, position, jd_clean = _extract_jd_info(raw, client)
        # Override with user-supplied values if given
        company = args.company or company
        position = args.position or position
        # Save clean JD for auditing
        jd_dir = Path(config.JD_DIR)
        jd_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_filename(f"{company}_{position}")
        jd_path = jd_dir / f"{stem}.txt"
        jd_path.write_text(jd_clean, encoding="utf-8")
        print(f"  JD saved : {jd_path}")
        _process_one(base_resume, jd_path, company, position, output_dir, client, cover=args.cover, link=args.url)
        print("\nDone.")
        return

    if args.batch:
        jd_dir = Path(config.JD_DIR)
        jd_files = sorted(jd_dir.glob("*.txt"))
        if not jd_files:
            print(f"ERROR: No .txt files found in {jd_dir}/")
            sys.exit(1)
        print(f"\nBatch mode: processing {len(jd_files)} job description(s) …")
        for jd_file in jd_files:
            # Expected filename convention: CompanyName_PositionTitle.txt
            parts = jd_file.stem.split("_", 1)
            company = parts[0] if len(parts) >= 1 else ""
            position = parts[1] if len(parts) >= 2 else jd_file.stem
            _process_one(base_resume, jd_file, company, position, output_dir, client, optimize=True, cover=args.cover)
    else:
        jd_path = Path(args.jd)
        company = args.company
        position = args.position
        # Auto-extract company/position from JD file if not supplied
        if not company or not position:
            print("  Extracting company / position from JD …")
            raw = jd_path.read_text(encoding="utf-8")
            c, p, _ = _extract_jd_info(raw, client)
            company = company or c
            position = position or p
        _process_one(
            base_resume,
            jd_path,
            company,
            position,
            output_dir,
            client,
            optimize=True,
            cover=args.cover,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
