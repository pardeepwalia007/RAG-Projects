from __future__ import annotations
# Suppresses logging and warnings from Docling for cleaner output.
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from docling.document_converter import DocumentConverter
import re
from pathlib import Path
import warnings
import logging

# 1. Suppress Python-level warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Silence the specific loggers for Docling/Docling-core
# These are the usual culprits for the "noisy" output
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("docling_core").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def clean_markdown(md_text: str) -> str:
    """Cleans markdown text by fixing escapes, symbols, and spacing for improved readability."""
    # 1. Fix Escape Characters (Crucial for SQL column names like sale_id)
    text = md_text.replace(r"\_", "_")
    
    # 2. Normalize Math Symbols for LLM/Code Generation
    text = text.replace("×", "*")
    
    # 3. Collapse multiple spaces/tabs into a single space
    # (Fixes: "This    analytical  system" -> "This analytical system")
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    # 4. Fix "Punctuation Gaps" (Fixes: "csv   dataset.    No" -> "csv dataset. No")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    
    # 5. Handle "Space-Collapse" (Just in case some sections were smushed)
    text = re.sub(r'([:.!?])([a-zA-Z])', r'\1 \2', text)
    
    # 6. Formatting: remove trailing whitespace and excessive newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def pdfs_to_markdown(
    pdf_paths: List[Path],
    output_dir: Path,
    *,
    min_md_chars: int = 50,
    overwrite: bool = True,
) -> Tuple[List[Path], List[Dict], bool]:
    """
    Converts PDFs to Markdown using Docling, with cleaning and error handling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()

    md_paths: List[Path] = []
    errors: List[Dict] = []

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        md_path = output_dir / f"{pdf_path.stem}.md"

        try:
            # if md exisist dont override
            if md_path.exists() and not overwrite:
                logger.info("Skipping (exists): %s", md_path)
                md_paths.append(md_path)
                continue

            logger.info("Converting → Markdown: %s", pdf_path)
            t0 = time.time()

            result = converter.convert(str(pdf_path))
            md_text = result.document.export_to_markdown()

            if len((md_text or "").strip()) < min_md_chars:
                raise ValueError("Markdown output too short/empty after conversion")

            # cleans and output md to md file
            clean_md = clean_markdown(md_text)
            md_path.write_text(clean_md, encoding="utf-8")

            dt = time.time() - t0
            logger.info("Saved: %s (%.2fs)", md_path, dt)

            md_paths.append(md_path)

        except Exception as e:
            logger.warning("Failed conversion: %s | %s", pdf_path, e)
            errors.append({"file": str(pdf_path), "error": str(e)})

    ok = len(md_paths) > 0
    return md_paths, errors, ok