"""Parse and search related-work references from research_landscape.md.

Extracts structured PaperReference objects from the markdown tables in
Section 7 ("Key Papers Reference List") and allows simple keyword search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaperReference:
    """A single paper reference extracted from the research landscape."""

    title: str
    authors: str
    year: int
    key_finding: str
    category: str


def load_related_work(landscape_path: Path) -> list[PaperReference]:
    """Parse research_landscape.md and extract paper references from tables.

    Looks for markdown tables with columns: Paper | Year | Key Finding,
    grouped under category headings (### headings inside section 7).
    """
    text = landscape_path.read_text()

    papers: list[PaperReference] = []

    # Find Section 7 onwards
    section7_match = re.search(r"^## 7\.", text, re.MULTILINE)
    if not section7_match:
        return papers

    section7_text = text[section7_match.start():]

    # Split by ### headings to get categories
    category_splits = re.split(r"^### (.+)$", section7_text, flags=re.MULTILINE)
    # category_splits: [preamble, cat_name, cat_body, cat_name, cat_body, ...]

    for i in range(1, len(category_splits), 2):
        category = category_splits[i].strip()
        body = category_splits[i + 1] if i + 1 < len(category_splits) else ""

        # Parse table rows: | Paper (Author) | Year | Key Finding |
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")]
            # cells[0] is empty (before first |), cells[-1] is empty (after last |)
            cells = [c for c in cells if c]
            if len(cells) < 3:
                continue
            # Skip header/separator rows
            if cells[0].lower().startswith("paper") or cells[0].startswith("-"):
                continue

            paper_cell = cells[0]
            year_cell = cells[1]
            finding_cell = cells[2]

            # Extract authors from parentheses
            authors_match = re.search(r"\(([^)]+)\)", paper_cell)
            authors = authors_match.group(1) if authors_match else ""

            # Title is everything before the parenthesized authors
            title = re.sub(r"\s*\([^)]*\)\s*$", "", paper_cell).strip()

            try:
                year = int(year_cell)
            except ValueError:
                continue

            papers.append(
                PaperReference(
                    title=title,
                    authors=authors,
                    year=year,
                    key_finding=finding_cell,
                    category=category,
                )
            )

    return papers


def find_relevant_papers(
    topic: str, papers: list[PaperReference]
) -> list[PaperReference]:
    """Simple keyword search across title and key_finding fields."""
    topic_lower = topic.lower()
    return [
        p
        for p in papers
        if topic_lower in p.title.lower() or topic_lower in p.key_finding.lower()
    ]
