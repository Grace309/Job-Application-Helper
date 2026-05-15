"""Use Claude to generate a tailored cover letter, then compile it to PDF via LaTeX."""

from __future__ import annotations

import json
import re
from pathlib import Path

from anthropic import Anthropic

from config import CLAUDE_MODEL


def _text_to_pdf_direct(cover_letter: str, resume: dict, pdf_path: Path) -> bool:
    """
    Generate a PDF from plain-text cover letter using only Python builtins.
    Uses PDF Type1 built-in fonts (Helvetica / Helvetica-Bold) — no dependencies needed.
    """

    def _pdf_escape(s: str) -> str:
        """Encode as latin-1 and escape for a PDF string literal."""
        s = s.encode("latin-1", errors="replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _wrap(text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    PAGE_W, PAGE_H = 612, 792   # US Letter in points
    ML, MT, MB = 72, 54, 54    # margins: left, top, bottom
    TEXT_W = PAGE_W - ML - ML  # 468 pts

    # Approximate chars/line for Helvetica (avg ~0.5 × pt per char)
    WRAP_BODY  = int(TEXT_W / (11 * 0.50))  # ~85
    WRAP_SMALL = int(TEXT_W / (10 * 0.50))  # ~93

    name     = resume.get("name", "")
    email    = resume.get("email", "")
    phone    = resume.get("phone", "")
    linkedin = resume.get("linkedin", "")

    # Build logical line list: (font_pt, bold, text)
    logical: list[tuple[int, bool, str]] = []
    if name:
        logical.append((13, True, name))
    contact = " · ".join(p for p in [email, phone, linkedin] if p)
    if contact:
        logical.append((10, False, contact))
    logical.append((11, False, ""))  # spacer after header

    for para in cover_letter.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for raw_line in para.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                logical.append((11, False, ""))
                continue
            for wl in _wrap(raw_line, WRAP_BODY):
                logical.append((11, False, wl))
        logical.append((11, False, ""))  # paragraph spacer

    # Paginate: each page is list of (x, y, pt, bold, text)
    LINE_H = {13: 19, 11: 15, 10: 13}
    pages: list[list[tuple]] = []
    page: list[tuple] = []
    y = PAGE_H - MT

    for pt, bold, text in logical:
        lh = LINE_H.get(pt, 15)
        if y - lh < MB and page:
            pages.append(page)
            page = []
            y = PAGE_H - MT
        page.append((ML, y, pt, bold, text))
        y -= lh
    if page:
        pages.append(page)
    if not pages:
        pages = [[]]

    # ── Build PDF binary ───────────────────────────────────────────────────────────
    n = len(pages)
    # Object IDs:
    #   1 = Catalog
    #   2 = Pages
    #   3 = /Helvetica font
    #   4 = /Helvetica-Bold font
    #   5 .. 4+n        = content streams
    #   5+n .. 4+2n     = page objects
    CATALOG_ID = 1
    PAGES_ID   = 2
    FONT_R_ID  = 3   # regular
    FONT_B_ID  = 4   # bold
    CS_START   = 5
    PG_START   = 5 + n

    obj: dict[int, bytes] = {}

    obj[FONT_R_ID] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    obj[FONT_B_ID] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"

    page_ids = []
    for i, page_lines in enumerate(pages):
        cs_id = CS_START + i
        pg_id = PG_START + i
        page_ids.append(pg_id)

        parts = ["BT\n"]
        for (x, y_pos, pt, bold, text) in page_lines:
            font = "F2" if bold else "F1"
            safe = _pdf_escape(text)
            parts.append(f"/{font} {pt} Tf 1 0 0 1 {x} {y_pos} Tm ({safe}) Tj\n")
        parts.append("ET\n")
        stream = "".join(parts).encode("latin-1", errors="replace")
        obj[cs_id] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"

        obj[pg_id] = (
            f"<< /Type /Page /Parent {PAGES_ID} 0 R "
            f"/MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Contents {cs_id} 0 R "
            f"/Resources << /Font << /F1 {FONT_R_ID} 0 R /F2 {FONT_B_ID} 0 R >> >> >>"
        ).encode()

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    obj[PAGES_ID]   = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    obj[CATALOG_ID] = f"<< /Type /Catalog /Pages {PAGES_ID} 0 R >>".encode()

    # Serialize
    max_id = max(obj.keys())
    buf = bytearray()
    buf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: dict[int, int] = {}
    for oid in range(1, max_id + 1):
        offsets[oid] = len(buf)
        buf.extend(f"{oid} 0 obj\n".encode())
        buf.extend(obj.get(oid, b"null"))
        buf.extend(b"\nendobj\n")

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {max_id + 1}\n".encode())
    buf.extend(b"0000000000 65535 f \n")          # free entry (20 bytes)
    for oid in range(1, max_id + 1):
        buf.extend(f"{offsets[oid]:010d} 00000 n \n".encode())  # 20 bytes

    buf.extend(
        f"trailer\n<< /Size {max_id + 1} /Root {CATALOG_ID} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )

    pdf_path.write_bytes(bytes(buf))
    return True


def generate_cover_letter_text(
    resume: dict,
    job_description: str,
    company: str,
    position: str,
    client: Anthropic,
) -> str:
    """Return a plain-text cover letter tailored to the JD."""

    prompt = f"""You are an expert career coach and professional writer.
Write a compelling, personalized cover letter for the following job application.

═══════════════════════════ JOB INFORMATION ═══════════════════════════
Company   : {company}
Position  : {position}

Job Description:
\"\"\"
{job_description}
\"\"\"

═══════════════════════════ CANDIDATE RESUME ═══════════════════════════
{json.dumps(resume, indent=2, ensure_ascii=False)}

═══════════════════════════ INSTRUCTIONS ════════════════════════════
Write a professional cover letter that:

1. **Opening paragraph** – Hook the reader. State the role and express genuine
   enthusiasm for {company} specifically (reference something real about them).
2. **Body (1–2 paragraphs)** – Connect 2–3 of the candidate's strongest, most
   relevant experiences/projects to the JD's key requirements. Be specific and
   use numbers from the resume where they exist.
3. **Closing paragraph** – Reiterate fit, express eagerness to discuss further,
   thank the reader.
4. **Tone** – Professional but warm. Avoid clichés like "I am writing to apply".
5. **Length** – 3–4 short paragraphs, ~300–380 words total.

ABSOLUTE RULES:
- Do NOT invent facts, experiences, or credentials not present in the resume.
- Do NOT fabricate metrics not already in the resume.
- Address it to "Hiring Manager" (no specific name).
- Include the candidate's name and contact info at the bottom signature.

Output ONLY the cover letter text — no extra commentary, no markdown fences.
Start directly with the date line or greeting.
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


def _text_to_latex(cover_letter: str, resume: dict) -> str:
    """Wrap plain-text cover letter in a clean LaTeX document."""
    name    = _escape_latex(resume.get("name", ""))
    email   = _escape_latex(resume.get("email", ""))
    phone   = _escape_latex(resume.get("phone", ""))
    linkedin = _escape_latex(resume.get("linkedin", ""))

    # Split into paragraphs and escape each one
    paragraphs = [p.strip() for p in cover_letter.split("\n\n") if p.strip()]
    body_lines = "\n\n".join(_escape_latex(p) for p in paragraphs)

    return rf"""\documentclass[letterpaper,11pt]{{article}}
\usepackage[top=0.75in,bottom=0.75in,left=1in,right=1in]{{geometry}}
\usepackage{{fontenc}}
\usepackage{{inputenc}}
\usepackage{{microtype}}
\usepackage{{parskip}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,urlcolor=blue,linkcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.8em}}

\begin{{document}}
\pagestyle{{empty}}

% Header
{{\large \textbf{{{name}}}}}\\[2pt]
{email} $\cdot$ {phone}\\
\href{{https://{resume.get("linkedin", "")}}}{{linkedin.com/in/{resume.get("linkedin","").split("/")[-1]}}}

\vspace{{1em}}

{body_lines}

\end{{document}}
"""


def build_cover_letter(
    resume: dict,
    job_description: str,
    company: str,
    position: str,
    output_dir: Path,
    stem: str,
    client: Anthropic,
) -> tuple[Path, Path | None]:
    """
    Generate a cover letter, save as .txt and .tex, compile to PDF.
    Returns (tex_path, pdf_path_or_None).
    """
    from src.compiler import compile_latex

    print("  Generating cover letter with Claude …")
    text = generate_cover_letter_text(resume, job_description, company, position, client)

    # Save plain text
    txt_path = output_dir / f"{stem}_cover_letter.txt"
    txt_path.write_text(text, encoding="utf-8")

    # Build and save LaTeX
    latex = _text_to_latex(text, resume)
    tex_path = output_dir / f"{stem}_cover_letter.tex"
    tex_path.write_text(latex, encoding="utf-8")

    # Try pdflatex first, then fall back to direct Python PDF generation
    pdf_path = compile_latex(tex_path, output_dir)
    if pdf_path:
        tex_path.unlink(missing_ok=True)
        txt_path.unlink(missing_ok=True)
        print(f"  Cover   : {pdf_path}")
    else:
        # Fallback: generate PDF directly with zero-dependency pure Python writer
        direct_pdf = output_dir / f"{stem}_cover_letter.pdf"
        if _text_to_pdf_direct(text, resume, direct_pdf):
            tex_path.unlink(missing_ok=True)
            txt_path.unlink(missing_ok=True)
            pdf_path = direct_pdf
            print(f"  Cover   : {pdf_path}")
        else:
            print(f"  Cover   : (compile manually – see {tex_path})")

    return tex_path, pdf_path
