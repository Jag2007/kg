from io import BytesIO
import json
import re
from collections import Counter

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - handled when PDF parsing is actually used
    fitz = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - handled when PDF parsing is actually used
    PdfReader = None


KEYWORD_STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "does",
    "during",
    "each",
    "from",
    "have",
    "into",
    "more",
    "other",
    "over",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def extract_pdf_pages(file_bytes: bytes) -> list[dict]:
    """Extract page-aware PDF text so chunks can keep page metadata."""
    pages: list[dict] = []
    if fitz is not None:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for index, page in enumerate(doc, start=1):
            text = page.get_text() or ""
            if text.strip():
                pages.append({"page": index, "text": text})
        return pages

    if PdfReader is not None:
        reader = PdfReader(BytesIO(file_bytes))
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": index, "text": text})
        return pages

    raise RuntimeError(
        "No PDF parser is available in the current environment. "
        "Install dependencies from requirements.txt before uploading PDFs."
    )
    return pages

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    return "\n".join(page["text"] for page in extract_pdf_pages(file_bytes))

def chunk_text(
    text: str,
    target_words: int = 180,
    min_words: int = 120,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into sentence-aware overlapping chunks so context is preserved better."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_sentences and current_words + sentence_words > max_words:
            chunks.append(" ".join(current_sentences).strip())

            overlap_sentences: list[str] = []
            overlap_count = 0
            for previous in reversed(current_sentences):
                previous_words = len(previous.split())
                if overlap_sentences and overlap_count + previous_words > overlap_words:
                    break
                overlap_sentences.insert(0, previous)
                overlap_count += previous_words

            current_sentences = list(overlap_sentences)
            current_words = overlap_count

        current_sentences.append(sentence)
        current_words += sentence_words

        if current_words >= target_words:
            chunks.append(" ".join(current_sentences).strip())

            overlap_sentences = []
            overlap_count = 0
            for previous in reversed(current_sentences):
                previous_words = len(previous.split())
                if overlap_sentences and overlap_count + previous_words > overlap_words:
                    break
                overlap_sentences.insert(0, previous)
                overlap_count += previous_words

            current_sentences = list(overlap_sentences)
            current_words = overlap_count

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    if len(chunks) >= 2:
        last_words = len(chunks[-1].split())
        if last_words < min_words:
            prev_words = len(chunks[-2].split())
            if prev_words + last_words <= max_words:
                chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
                chunks.pop()

    deduped_chunks: list[str] = []
    seen = set()
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_chunks.append(chunk.strip())

    return deduped_chunks


def build_hybrid_chunks(
    pages: list[dict],
    target_words: int = 550,
    min_words: int = 380,
    max_words: int = 700,
    overlap_ratio: float = 0.20,
) -> list[dict]:
    """
    Build rich JSON chunks using a hybrid strategy:
    - page hierarchy
    - heading/section awareness
    - paragraph and sentence boundaries
    - sliding-window overlap
    """
    overlap_words = max(1, int(target_words * min(max(overlap_ratio, 0.15), 0.25)))
    sentence_units = _build_sentence_units(pages)
    if not sentence_units:
        return []

    chunks: list[dict] = []
    current_units: list[dict] = []
    current_words = 0

    def flush_current() -> None:
        if not current_units:
            return
        chunk = _make_chunk(len(chunks) + 1, current_units)
        if chunk["text"]:
            chunks.append(chunk)

    for unit in sentence_units:
        unit_words = len(unit["text"].split())
        if current_units and current_words + unit_words > max_words:
            flush_current()
            current_units, current_words = _overlap_units(current_units, overlap_words)

        current_units.append(unit)
        current_words += unit_words

        if current_words >= target_words:
            flush_current()
            current_units, current_words = _overlap_units(current_units, overlap_words)

    if current_units:
        flush_current()

    if len(chunks) >= 2 and len(chunks[-1]["text"].split()) < min_words:
        previous_words = len(chunks[-2]["text"].split())
        last_words = len(chunks[-1]["text"].split())
        if previous_words + last_words <= max_words:
            merged_units = chunks[-2].get("_units", []) + chunks[-1].get("_units", [])
            chunks[-2] = _make_chunk(len(chunks) - 1, merged_units)
            chunks.pop()

    clean_chunks: list[dict] = []
    seen = set()
    for chunk in chunks:
        text_key = re.sub(r"\s+", " ", chunk["text"]).strip().lower()
        if not text_key or text_key in seen:
            continue
        seen.add(text_key)
        chunk.pop("_units", None)
        chunk["id"] = f"chunk_{len(clean_chunks) + 1}"
        clean_chunks.append(chunk)

    return clean_chunks


def _build_sentence_units(pages: list[dict]) -> list[dict]:
    units: list[dict] = []
    current_section = "Document"

    for page in pages:
        page_number = int(page.get("page", 0) or 0)
        text = str(page.get("text", ""))
        for block in _split_paragraph_blocks(text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) == 1 and _looks_like_heading(lines[0]):
                current_section = lines[0].strip()
                continue

            paragraph = re.sub(r"\s+", " ", block).strip()
            if not paragraph:
                continue

            for sentence in split_into_sentences(paragraph):
                units.append(
                    {
                        "text": sentence,
                        "section": current_section,
                        "page": page_number,
                    }
                )

    return units


def _split_paragraph_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", normalized)
    if len(blocks) > 1:
        return [block.strip() for block in blocks if block.strip()]

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if _looks_like_heading(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 90:
        return False
    if text.endswith((".", ",", ";")):
        return False
    words = text.split()
    if len(words) > 10:
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
        return True
    alpha = re.sub(r"[^A-Za-z]", "", text)
    if alpha and text.upper() == text and len(alpha) >= 4:
        return True
    title_words = sum(1 for word in words if word[:1].isupper())
    return bool(words) and title_words >= max(1, len(words) - 1)


def _overlap_units(units: list[dict], overlap_words: int) -> tuple[list[dict], int]:
    overlap: list[dict] = []
    word_count = 0
    for unit in reversed(units):
        unit_words = len(unit["text"].split())
        if overlap and word_count + unit_words > overlap_words:
            break
        overlap.insert(0, unit)
        word_count += unit_words
    return overlap, word_count


def _make_chunk(index: int, units: list[dict]) -> dict:
    text = " ".join(unit["text"] for unit in units).strip()
    sections = [unit.get("section", "Document") for unit in units if unit.get("section")]
    pages = [int(unit.get("page", 0) or 0) for unit in units if unit.get("page")]
    section = _most_common(sections) or "Document"
    page = min(pages) if pages else None
    return {
        "id": f"chunk_{index}",
        "text": text,
        "summary": summarize_text(text),
        "keywords": extract_keywords(text),
        "section": section,
        "page": page,
        "_units": list(units),
    }


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", str(text or ""))
        if token.lower() not in KEYWORD_STOP_WORDS and len(token) > 2
    ]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def summarize_text(text: str, max_sentences: int = 2) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences]).strip()

def split_into_sentences(text: str) -> list[str]:
    """Split text into a list of sentences cleanly. (Legacy)"""
    normalized_text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
    # Clean and filter out very short results
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def strip_code_fences(text: str) -> str:
    """Remove common markdown code fences around JSON/Cypher outputs."""
    s = text.strip()
    if s.startswith("```"):
        # ```json\n...\n``` or ```\n...\n```
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3].strip()
    return s.strip()


def extract_first_json(text: str) -> str:
    """
    Extract the first JSON array/object from a string.
    Returns the raw JSON substring (not parsed).
    """
    if not text:
        raise ValueError("Empty LLM response")

    s = strip_code_fences(text)

    # Prefer arrays, then objects.
    array_start = s.find("[")
    object_start = s.find("{")

    if array_start != -1 and (object_start == -1 or array_start < object_start):
        start = array_start
        end = s.rfind("]")
    else:
        start = object_start
        end = s.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON in LLM response")
    return s[start : end + 1]


def recover_json_objects(text: str) -> list[dict]:
    """
    Recover complete JSON objects from a possibly truncated JSON array.
    This is a safety net for LLM responses that stop mid-array.
    """
    if not text:
        return []

    s = strip_code_fences(text)
    objects: list[dict] = []

    for match in re.finditer(r"\{[^{}]*\}", s, flags=re.DOTALL):
        try:
            item = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(item, dict):
            objects.append(item)

    return objects


def normalise_relation_for_llm(relation: str) -> str:
    """Normalize relation to UPPERCASE_WITH_UNDERSCORES for extraction and storage."""
    rel = str(relation or "").strip().upper()
    rel = rel.replace(" ", "_").replace("-", "_")
    rel = re.sub(r"[^A-Z0-9_]", "", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel


_RELATION_CANON = {
    "WAS_FOUNDED_BY": "FOUNDED_BY",
    "IS_FOUNDED_BY": "FOUNDED_BY",
    "WAS_FOUNDED": "FOUNDED",
    "IS_FOUNDED": "FOUNDED",
    "CONSISTS_OF": "INCLUDES",
    "COMPRISED_OF": "INCLUDES",
    "COMPOSED_OF": "INCLUDES",
    "CONTAINS": "INCLUDES",
    "CONTAIN": "INCLUDES",
    "ALSO_KNOWN_AS": "KNOWN_AS",
    "IS_KNOWN_AS": "KNOWN_AS",
    "IS_ALSO_KNOWN_AS": "KNOWN_AS",
    "WORKS_FOR": "EMPLOYED_BY",
    "WORK_FOR": "EMPLOYED_BY",
    "IS_EMPLOYED_BY": "EMPLOYED_BY",
    "WAS_EMPLOYED_BY": "EMPLOYED_BY",
    "IS_LOCATED_IN": "LOCATED_IN",
    "WAS_LOCATED_IN": "LOCATED_IN",
    "IS_BASED_IN": "BASED_IN",
    "WAS_BASED_IN": "BASED_IN",
    "WAS_ADOPTED_ON": "ADOPTED_ON",
    "IS_ADOPTED_ON": "ADOPTED_ON",
    "WAS_DRAFTED_BY": "DRAFTED_BY",
    "IS_DRAFTED_BY": "DRAFTED_BY",
    "IS_PART_OF": "PART_OF",
    "WAS_PART_OF": "PART_OF",
    "IS_MEMBER_OF": "MEMBER_OF",
    "WAS_MEMBER_OF": "MEMBER_OF",
    "IS_CAPITAL_OF": "CAPITAL_OF",
    "HAS_CAPITAL_CITY": "HAS_CAPITAL",
    "CAPITAL_CITY_OF": "CAPITAL_OF",
    "WAS_BORN_IN": "BORN_IN",
    "IS_BORN_IN": "BORN_IN",
    "DIED_IN": "DIED_IN",
    "WAS_DIED_IN": "DIED_IN",
    "IS_KNOWN_FOR": "KNOWN_FOR",
    "WAS_KNOWN_FOR": "KNOWN_FOR",
}


def normalise_relation_for_storage(rel_type: str) -> str:
    """
    Normalize relation labels into a stable uppercase token for storage and retrieval.
    """
    rel = normalise_relation_for_llm(rel_type)
    if not rel:
        return "RELATED"
    if re.match(r"^[0-9]", rel):
        rel = f"R_{rel}"
    rel = _RELATION_CANON.get(rel, rel)
    return rel[:60]


def load_env() -> None:
    """Backwards-compatible helper for older modules."""
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)


def normalise_cypher_response(text: str) -> str:
    """Strip markdown fences and keep the raw Cypher text."""
    return strip_code_fences(text)
