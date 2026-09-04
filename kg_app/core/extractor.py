import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime when provider is used
    OpenAI = None

from kg_app.core.utils import (
    extract_first_json,
    normalise_relation_for_llm,
    recover_json_objects,
    split_into_sentences,
)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

NOISY_RELATIONS = {
    "IS",
    "HAS",
    "ARE",
    "ARE_COMMON",
    "OLDEST",
    "MOST_DIVERSE",
    "VARY_BY",
    "VARY_WIDELY",
    "SPREADS",
    "EXPRESSES",
    "EXPRESS",
    "BASED_ON",
    "CONTINUES_TO_EVOLVE",
    "IMPORTANT",
    "POPULAR",
    "FAMOUS",
    "KNOWN_FOR",
    "RELATED_TO",
    "ASSOCIATED_WITH",
    "PLAYS",
    "PLAYS_ROLE",
    "CONSIDERED",
    "DESCRIBED_AS",
    "CAN_BE",
    "MAY_BE",
    "EXISTS",
    "INVOLVED_IN",
    "PART_OF_A",
    "SEEN_AS",
    "LINKED_TO",
}

INVALID_OBJECTS = {
    "world",
    "company",
    "organization",
    "person",
    "culture",
    "society",
    "not specified",
    "not_specified",
    "unknown",
    "none",
    "null",
    "various",
    "many",
    "several",
    "things",
    "stuff",
    "examples",
    "type",
    "types",
    "something",
    "anything",
    "everything",
    "others",
    "other",
    "more",
    "much",
    "some",
}

INVALID_SUBJECTS = {
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "we",
    "i",
    "you",
}

_EXTRACTION_CACHE: dict[str, list[dict]] = {}
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_LOGGED_PROVIDER_CONFIGS: set[tuple[str, str]] = set()
DEBUG_LLM_OUTPUT = os.environ.get("KG_DEBUG_LLM_OUTPUT", "0").strip().lower() in {"1", "true", "yes"}
_TRAILING_ARTICLES = re.compile(r"\b(its|their|his|her|the|an|a)\s*$", re.IGNORECASE)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


VERBOSE_BACKEND_LOGS = _env_flag("KG_VERBOSE_BACKEND_LOGS", "1")


def _backend_log(message: str) -> None:
    if VERBOSE_BACKEND_LOGS:
        print(f"[extractor] {message}", flush=True)


def _get_provider() -> str:
    if os.environ.get("HEBBRIX_API_KEY"):
        return "hebbrix"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "unknown"


def _provider_label() -> str:
    provider = _get_provider()
    if provider == "hebbrix":
        return "Hebbrix"
    if provider == "openai":
        return "OpenAI-compatible"
    return "LLM"


def _current_base_url() -> str:
    if _get_provider() == "hebbrix":
        return os.environ.get("HEBBRIX_BASE_URL", "https://api.hebbrix.com/v1").strip()
    return os.environ.get("OPENAI_BASE_URL", "").strip()


def _get_client():
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required for Hebbrix/OpenAI-compatible providers. "
            "Install dependencies from requirements.txt."
        )
    if os.environ.get("HEBBRIX_API_KEY"):
        config = ("hebbrix", _current_base_url() or "default")
        if config not in _LOGGED_PROVIDER_CONFIGS:
            _backend_log(f"Using Hebbrix provider with base URL {config[1]}.")
            _LOGGED_PROVIDER_CONFIGS.add(config)
        return OpenAI(
            api_key=os.environ.get("HEBBRIX_API_KEY"),
            base_url=os.environ.get("HEBBRIX_BASE_URL", "https://api.hebbrix.com/v1"),
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        config = ("openai", base_url or "default")
        if config not in _LOGGED_PROVIDER_CONFIGS:
            _backend_log(f"Using OpenAI-compatible provider with base URL {config[1]}.")
            _LOGGED_PROVIDER_CONFIGS.add(config)
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    raise RuntimeError("Set HEBBRIX_API_KEY or OPENAI_API_KEY in .env.")


def _extract_retry_delay(error: Exception) -> float:
    message = str(error)
    match = re.search(r"try again in ([0-9.]+)s", message, re.IGNORECASE)
    if match:
        try:
            return max(1.0, float(match.group(1)) + 1.0)
        except ValueError:
            pass
    return float(os.environ.get("LLM_RETRY_DELAY_SECONDS", os.environ.get("GROQ_RETRY_DELAY_SECONDS", "12")))


def _llm_error_details(error: Exception, *, model: str) -> tuple[bool, str]:
    """
    Return (retryable, user_facing_reason) for terminal logging and fast-fail control.
    """
    message = str(error or "").strip()
    lowered = message.lower()
    provider = _provider_label()
    base_url = _current_base_url() or "default"

    if any(code in lowered for code in ("401", "unauthorized", "invalid api key", "incorrect api key", "authentication")):
        return False, (
            f"{provider} authentication failed. Check your API key in .env."
        )

    if any(code in lowered for code in ("403", "forbidden", "permission")):
        return False, (
            f"{provider} rejected the request. Check key permissions and account access."
        )

    if any(code in lowered for code in ("404", "model_not_found", "unknown model", "does not exist")):
        return False, (
            f"{provider} model '{model}' was not found. Check the model name and base URL ({base_url})."
        )

    if "429" in lowered or "rate limit" in lowered:
        return True, f"{provider} rate limit hit."

    if any(code in lowered for code in ("413", "request too large", "context length", "maximum context length")):
        return False, (
            f"{provider} rejected the request as too large for model '{model}'. Reduce chunk/prompt size."
        )

    if any(code in lowered for code in ("dns", "name or service not known", "nodename", "connection refused", "timed out", "timeout")):
        return True, f"{provider} network/connectivity issue while calling {base_url}."

    if any(code in lowered for code in ("400", "bad request")):
        return False, (
            f"{provider} rejected the request as invalid. Check model '{model}', base URL ({base_url}), and payload format."
        )

    return True, f"{provider} request failed."


def _call_llm_with_retry(client, *, model: str, prompt: str):
    global _LAST_REQUEST_AT

    max_retries = max(1, int(os.environ.get("LLM_MAX_RETRIES", os.environ.get("GROQ_MAX_RETRIES", "4"))))
    min_interval = max(0.0, float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "1.5"))))
    max_tokens = max(128, int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "1024"))))

    last_error = None
    for attempt in range(max_retries):
        with _REQUEST_LOCK:
            now = time.monotonic()
            wait_time = min_interval - (now - _LAST_REQUEST_AT)
            if wait_time > 0:
                time.sleep(wait_time)

            try:
                _backend_log(
                    f"Sending LLM request with model '{model}' "
                    f"(attempt {attempt + 1}/{max_retries}, prompt_chars={len(prompt)})."
                )
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                _LAST_REQUEST_AT = time.monotonic()
                return completion
            except Exception as error:
                _LAST_REQUEST_AT = time.monotonic()
                last_error = error
                retryable, reason = _llm_error_details(error, model=model)
                if not retryable:
                    print(f"{reason} Full error: {error}", flush=True)
                    raise
                delay = _extract_retry_delay(error)
                print(
                    f"{reason} Attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {delay:.1f}s before retrying.",
                    flush=True,
                )

        if attempt < max_retries - 1:
            time.sleep(delay)

    raise last_error if last_error else RuntimeError("LLM extraction failed.")


def precompute_chunk_metadata(chunks: list[dict]) -> list[dict]:
    """
    Enrich chunk JSON with LLM-generated summaries and keywords in one ingestion-time call.
    If the LLM call fails, the heuristic metadata already on each chunk is kept.
    """
    if not chunks:
        return []

    max_chunks = max(1, int(os.environ.get("KG_METADATA_MAX_CHUNKS", "30")))
    if os.environ.get("HEBBRIX_API_KEY"):
        model = os.environ.get("HEBBRIX_METADATA_MODEL", os.environ.get("HEBBRIX_TRIPLE_MODEL", "gpt-5-nano"))
    else:
        model = os.environ.get("OPENAI_METADATA_MODEL", os.environ.get("OPENAI_TRIPLE_MODEL", "gpt-4.1-mini"))
    _backend_log(
        f"Starting chunk metadata enrichment for {len(chunks)} chunks with model '{model}'."
    )
    payload = [
        {
            "id": chunk.get("id", f"chunk_{index + 1}"),
            "text": str(chunk.get("text", ""))[:2400],
        }
        for index, chunk in enumerate(chunks[:max_chunks])
        if str(chunk.get("text", "")).strip()
    ]
    if not payload:
        return chunks

    prompt = f"""
Create retrieval metadata for these PDF chunks.

STRICT RULES:
- Output ONLY valid JSON
- Return a JSON list
- Each item must be:
  {{"id": "chunk_1", "summary": "...", "keywords": ["...", "..."]}}
- Summary must be one short sentence
- Keywords must be 3 to 8 important topic words or phrases
- Do not invent facts

Chunks:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        client = _get_client()
        completion = _call_llm_with_retry(client, model=model, prompt=prompt)
        content = completion.choices[0].message.content or ""
        try:
            metadata = json.loads(extract_first_json(content))
        except Exception:
            _backend_log("Metadata JSON parse failed, attempting recovery from partial objects.")
            metadata = recover_json_objects(content)
        if not isinstance(metadata, list):
            return chunks

        by_id = {
            str(item.get("id", "")).strip(): item
            for item in metadata
            if isinstance(item, dict) and item.get("id")
        }
        enriched: list[dict] = []
        for chunk in chunks:
            updated = dict(chunk)
            item = by_id.get(str(chunk.get("id", "")).strip())
            if item:
                summary = str(item.get("summary", "")).strip()
                keywords = item.get("keywords", [])
                if summary:
                    updated["summary"] = summary
                if isinstance(keywords, list):
                    cleaned_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
                    if cleaned_keywords:
                        updated["keywords"] = cleaned_keywords[:8]
            enriched.append(updated)
        _backend_log(f"Metadata enrichment complete for {len(enriched)} chunks.")
        return enriched
    except Exception as error:
        print(f"Chunk metadata enrichment skipped: {error}", flush=True)
        return chunks


def _split_compound_entity(value: str) -> list[str]:
    """Split simple conjunction/list mentions into separate entity values."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return []

    # Keep short phrases intact; they are often proper nouns rather than lists.
    if len(text.split()) <= 3:
        return [text]

    if not re.search(r",|\band\b|\bor\b", text, flags=re.IGNORECASE):
        return [text]

    parts = [
        item.strip(" .;:")
        for item in re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", text, flags=re.IGNORECASE)
        if item.strip(" .;:")
    ]
    return parts or [text]


def _clean_context_phrase(value: str) -> str:
    """Clean a phrase before reusing it as a subject/object in context-linked triples."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^(the|a|an|these|those|this|that)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^for\s+(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(is|are|was|were|has|have|had)$", "", text, flags=re.IGNORECASE)
    return text.strip(" .;:")


def _infer_relation_from_phrase(phrase: str) -> str:
    """Map simple natural-language predicates to stable relation labels."""
    text = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
    relation_map = {
        "adopted on": "ADOPTED_ON",
        "was adopted on": "ADOPTED_ON",
        "does not allow for": "DOES_NOT_ALLOW",
        "does not allow": "DOES_NOT_ALLOW",
        "do not allow": "DOES_NOT_ALLOW",
        "helps to protect": "PROTECTS",
        "helps protect": "PROTECTS",
        "adopted": "ADOPTED",
        "drafted": "DRAFTED",
        "includes": "INCLUDES",
        "include": "INCLUDES",
        "contains": "INCLUDES",
        "consists of": "INCLUDES",
        "refers to": "REFERS_TO",
        "guarantees": "GUARANTEES",
        "protects": "PROTECTS",
        "reflects": "REFLECTS",
        "celebrates": "CELEBRATES",
        "practices": "PRACTICES",
        "speaks": "SPEAKS",
        "uses": "USES",
        "located in": "LOCATED_IN",
        "based in": "BASED_IN",
        "convened in": "CONVENED_IN",
        "member of": "MEMBER_OF",
    }
    for key, relation in relation_map.items():
        if key in text:
            return relation
    return ""


def _extract_direct_statement_triples(sentence: str, fallback_subject: str = "") -> list[dict]:
    """Extract direct one-sentence triples and resolve nearby pronoun subjects."""
    text = re.sub(r"\s+", " ", str(sentence or "").strip())
    if not text:
        return []

    patterns = [
        r"^(?P<subject>.+?)\s+(?P<relation>does not allow for|does not allow|do not allow|helps to protect|helps protect|was adopted on|adopted on|convened in|located in|based in|member of|refers to|consists of|includes|include|contains|drafted|adopted|guarantees|protects|reflects|celebrates|practices|speaks|uses)\s+(?P<object>.+?)\.?$",
        r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?P<relation>located in|based in|member of)\s+(?P<object>.+?)\.?$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        subject = _clean_context_phrase(match.group("subject"))
        relation = _infer_relation_from_phrase(match.group("relation"))
        object_text = _clean_context_phrase(match.group("object"))

        if subject.lower() in INVALID_SUBJECTS and fallback_subject:
            subject = fallback_subject

        if not subject or not relation or not object_text:
            return []

        return [
            {
                "subject": subject,
                "relation": relation,
                "object": _clean_context_phrase(object_part),
            }
            for object_part in _split_compound_entity(object_text)
            if _clean_context_phrase(object_part)
        ]

    return []


def _extract_interlinked_context_triples(text: str) -> list[dict]:
    """Recover short-range context links such as 'There are X. These are A, B and C.'"""
    context_subject = ""
    list_subject = ""
    triples: list[dict] = []

    for sentence in split_into_sentences(text):
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue

        intro_match = re.search(
            r"(?:there\s+are|there\s+is)\s+(?P<subject>.+?)\.?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if intro_match:
            list_subject = _clean_context_phrase(intro_match.group("subject"))
            context_subject = list_subject
            continue

        followup_match = re.match(
            r"^(?:these|those|they)(?:\s+are|\s+include)?\s+(?P<object>.+?)\.?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if followup_match and list_subject:
            for object_part in _split_compound_entity(followup_match.group("object")):
                cleaned_object = _clean_context_phrase(object_part)
                if cleaned_object:
                    triples.append(
                        {
                            "subject": list_subject,
                            "relation": "INCLUDES",
                            "object": cleaned_object,
                        }
                    )
            context_subject = list_subject
            continue

        direct_triples = _extract_direct_statement_triples(normalized, context_subject)
        if direct_triples:
            triples.extend(direct_triples)
            context_subject = direct_triples[-1]["subject"]
            list_subject = context_subject
            continue

        if not re.match(r"^(it|they|this|that|these|those)\b", normalized, flags=re.IGNORECASE):
            context_subject = ""
            list_subject = ""

    return triples

def clean_and_validate_triples(triples: list) -> list:
    """Validate and clean the LLM-extracted triples."""
    valid_triples = []

    seen = set()
    
    if not isinstance(triples, list):
        return []
        
    for t in triples:
        if not isinstance(t, dict):
            continue
            
        # Normalize minimal fields as requested.
        subj = str(t.get("subject", "")).strip()
        rel = str(t.get("relation", "")).strip()
        obj = str(t.get("object", "")).strip()

        # Only accept triples when required fields exist and are non-empty.
        if not subj or not rel or not obj:
            continue

        if subj.strip().lower() in INVALID_SUBJECTS:
            continue
            
        # Reject subject == object
        if subj.lower() == obj.lower():
            continue

        # Normalize: strip spaces + uppercase + underscore format.
        rel_clean = normalise_relation_for_llm(rel)
        if not rel_clean:
            continue

        if rel_clean in NOISY_RELATIONS:
            continue

        # Enforce "1–3 words" relations (heuristic: underscore-separated parts).
        parts = [p for p in rel_clean.split("_") if p]
        if len(parts) < 1 or len(parts) > 3:
            continue

        if obj.strip().lower() in INVALID_OBJECTS:
            continue

        if len(obj.split()) > 10:
            continue

        if len(subj.split()) > 8:
            continue

        if _TRAILING_ARTICLES.search(obj.strip()):
            continue
            
        for subject_part in _split_compound_entity(subj):
            subject_clean = _clean_context_phrase(subject_part)
            if not subject_clean or subject_clean.lower() in INVALID_SUBJECTS:
                continue
            if len(subject_clean) < 2:
                continue

            for object_part in _split_compound_entity(obj):
                object_clean = _clean_context_phrase(object_part)
                if not object_clean or object_clean.lower() in INVALID_OBJECTS:
                    continue
                if len(object_clean) < 2:
                    continue

                key = (subject_clean.lower(), rel_clean.lower(), object_clean.lower())
                if key in seen:
                    continue
                seen.add(key)

                valid_triples.append(
                    {
                        "subject": subject_clean,
                        "relation": rel_clean,
                        "object": object_clean,
                    }
                )
        
    return valid_triples


def extract_triples_fallback(text: str) -> list[dict]:
    """
    Lightweight local fallback when the remote LLM is unavailable.
    This keeps the upload usable on provider failures by recovering
    simple direct/contextual triples without any API call.
    """
    source = str(text or "").strip()
    if not source:
        return []

    heuristic_triples: list[dict] = []
    for sentence in split_into_sentences(source):
        heuristic_triples.extend(_extract_direct_statement_triples(sentence))
    heuristic_triples.extend(_extract_interlinked_context_triples(source))
    cleaned = clean_and_validate_triples(heuristic_triples)
    if cleaned:
        _backend_log(f"Fallback extraction recovered {len(cleaned)} local triples.")
    return cleaned


def extract_triples_llm(chunk: str) -> list:
    """Extract triples from one chunk with strict JSON parsing and per-chunk caching."""
    text = chunk.strip()
    if not text:
        return []

    cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if cache_key in _EXTRACTION_CACHE:
        _backend_log("Triple extraction cache hit.")
        return list(_EXTRACTION_CACHE[cache_key])

    if os.environ.get("HEBBRIX_API_KEY"):
        model = os.environ.get("HEBBRIX_TRIPLE_MODEL", "gpt-5-nano")
    else:
        model = os.environ.get("OPENAI_TRIPLE_MODEL", "gpt-4.1-mini")
    _backend_log(
        f"Starting triple extraction with model '{model}' for chunk of {len(text.split())} words."
    )
    client = _get_client()

    def _attempt(prompt: str) -> tuple[list, str]:
        completion = _call_llm_with_retry(client, model=model, prompt=prompt)

        response = completion.choices[0].message.content or ""
        if DEBUG_LLM_OUTPUT:
            print("RAW LLM OUTPUT:", response[:500])

        # Fallback trigger: empty response.
        if not response.strip():
            return [], "empty"

        try:
            json_blob = extract_first_json(response)
            triples = json.loads(json_blob)
        except Exception:
            triples = recover_json_objects(response)
            if not triples:
                # If parsing still fails -> skip this chunk without crashing the full upload.
                _backend_log("LLM response could not be parsed into JSON triples.")
                return [], "parse_failed"

        cleaned = clean_and_validate_triples(triples)
        if not cleaned:
            _backend_log("LLM response parsed, but no valid triples remained after cleaning.")
            return [], "empty"

        _backend_log(f"LLM extraction produced {len(cleaned)} validated triples.")
        return cleaned, "ok"

    base_prompt = f"""You are an expert knowledge extraction system.

Extract factual knowledge graph triples from the text.

Return ONLY valid JSON.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Rules:
- Extract every clearly stated factual triple.
- Aim for 30 or more triples when the text supports it.
- Do not stop early. Extract until you have covered every fact in the text.
- Keep subject and object short and clean.
- Relation must be 1-3 words.
- Use UPPERCASE_WITH_UNDERSCORES.
- Preserve paragraph context. If a later sentence uses it, they, this, these, or those, resolve it from the nearest previous entity in the same paragraph.
- Preserve list/conjunction structure. If one subject relates to A, B, and C, output one triple per object while repeating the same subject and relation.
- Prefer useful factual relations such as LOCATED_IN, WEARS, CELEBRATES, SPEAKS, INCLUDES, PRACTICES, USES, REFLECTS, GUARANTEES, PROTECTS, DRAFTED, ADOPTED_ON, CONVENED_IN, MEMBER_OF.
- Avoid vague predicate wording like PLAYS, HAS, IS, ARE, IMPORTANT_ROLE, or very long relation names.
- Skip weak descriptive relations like OLDEST, MOST_DIVERSE, IMPORTANT, KNOWN_FOR.
- Skip incomplete objects or trailing phrases.
- Preserve meaningful document facts, but do not invent relations that are not present in the text.
- Do not explain anything.
- If there are no clear factual triples, return [].

Text:
{text}
"""

    fallback_prompt = f"""Extract knowledge graph triples from the text.

Return ONLY valid JSON.

Format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Text:
{text}
"""

    triples, status = _attempt(base_prompt)
    if status == "empty":
        triples, _ = _attempt(fallback_prompt)

    context_triples = clean_and_validate_triples(_extract_interlinked_context_triples(text))
    if context_triples:
        merged_triples: list[dict] = []
        seen = set()
        for triple in list(triples) + context_triples:
            key = (
                triple["subject"].strip().lower(),
                triple["relation"].strip().lower(),
                triple["object"].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)
        triples = merged_triples

    if DEBUG_LLM_OUTPUT:
        print("CLEANED TRIPLES JSON:", json.dumps(triples, ensure_ascii=False))

    _EXTRACTION_CACHE[cache_key] = list(triples)
    _backend_log(f"Triple extraction finished with {len(triples)} final triples.")
    return list(triples)


# Backwards-compatible aliases while older imports are still being cleaned up.
_call_groq_with_retry = _call_llm_with_retry
extract_triples_groq = extract_triples_llm
