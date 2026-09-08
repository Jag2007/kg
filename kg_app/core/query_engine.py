import re
from difflib import SequenceMatcher

from kg_app.db.graph import GraphStore
from kg_app.core.embeddings import cosine_similarity, embed_text
from kg_app.core.utils import split_into_sentences
from kg_app.state.chunk_store import get_document_chunks

HYBRID_WEIGHTS = {
    "embedding": 0.45,
    "keyword": 0.25,
    "fuzzy": 0.1,
    "graph": 0.2,
}
MIN_HYBRID_SCORE = 0.18

STOP_WORDS = {
    "a",
    "an",
    "about",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "know",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "you",
    "your",
    "pdf",
    "document",
    "file",
    "form",
}

QUESTION_SYNONYMS = {
    "founder": ["founded", "founded by", "established by", "created by"],
    "location": ["located", "situated", "based", "found in", "position"],
    "member": ["members", "membership", "part of", "belongs to"],
    "leader": ["head", "chief", "director", "president", "chairman"],
    "author": ["written by", "authored by", "wrote", "creator"],
    "date": ["when", "year", "time", "period", "era"],
    "cause": ["caused by", "reason", "why", "due to", "result of"],
    "result": ["outcome", "effect", "consequence", "led to", "produced"],
    "purpose": ["goal", "aim", "objective", "intended for", "designed for"],
    "part": ["component", "element", "section", "portion", "division"],
    "capital": ["capital city", "seat of government", "headquarters"],
    "owner": ["owned by", "belongs to", "property of", "controlled by"],
}

GENERIC_NODE_PATTERNS = [
    "what do you know about",
    "tell me about",
    "what are",
    "who is",
    "who was",
    "what is",
    "describe",
    "explain",
    "give me information about",
    "give me details about",
    "what can you tell me about",
    "information about",
]

RELATION_HINTS = {
    "include": {"INCLUDES"}, "includes": {"INCLUDES"},
    "contain": {"INCLUDES"}, "contains": {"INCLUDES"},
    "consist": {"INCLUDES"}, "consists": {"INCLUDES"},
    "comprise": {"INCLUDES"}, "comprises": {"INCLUDES"},
    "capital": {"HAS_CAPITAL", "CAPITAL", "CAPITAL_OF"},
    "located": {"LOCATED_IN", "BASED_IN"},
    "location": {"LOCATED_IN", "BASED_IN"},
    "based": {"LOCATED_IN", "BASED_IN"},
    "situated": {"LOCATED_IN"},
    "position": {"LOCATED_IN"},
    "headquartered": {"BASED_IN"},
    "found": {"FOUNDED", "FOUNDED_BY"}, "founded": {"FOUNDED", "FOUNDED_BY"},
    "create": {"FOUNDED", "FOUNDED_BY"}, "created": {"FOUNDED", "FOUNDED_BY"},
    "establish": {"FOUNDED", "FOUNDED_BY"}, "established": {"FOUNDED", "FOUNDED_BY"},
    "invent": {"INVENTED_BY"}, "invented": {"INVENTED_BY"},
    "discover": {"DISCOVERED_BY"}, "discovered": {"DISCOVERED_BY"},
    "write": {"WRITTEN_BY"}, "wrote": {"WRITTEN_BY"}, "written": {"WRITTEN_BY"},
    "author": {"WRITTEN_BY", "AUTHORED_BY"}, "authored": {"WRITTEN_BY", "AUTHORED_BY"},
    "publish": {"PUBLISHED_IN", "PUBLISHED_BY"}, "published": {"PUBLISHED_IN", "PUBLISHED_BY"},
    "start": {"STARTED", "STARTED_IN", "STARTED_ON", "BEGAN", "FOUNDED", "CONVENED_IN"},
    "started": {"STARTED", "STARTED_IN", "STARTED_ON", "BEGAN", "FOUNDED", "CONVENED_IN"},
    "begin": {"STARTED", "BEGAN", "BEGAN_IN"}, "began": {"STARTED", "BEGAN", "BEGAN_IN"},
    "born": {"BORN_IN"}, "birth": {"BORN_IN"},
    "die": {"DIED_IN"}, "died": {"DIED_IN"}, "death": {"DIED_IN"},
    "father": {"KNOWN_AS", "FOUNDED_BY"}, "known": {"KNOWN_AS"},
    "called": {"KNOWN_AS"}, "named": {"KNOWN_AS"},
    "member": {"MEMBER_OF"}, "part": {"PART_OF"},
    "leader": {"LED_BY", "HEADED_BY"}, "head": {"LED_BY", "HEADED_BY"},
    "employ": {"EMPLOYED_BY"}, "employed": {"EMPLOYED_BY"},
    "work": {"EMPLOYED_BY"}, "works": {"EMPLOYED_BY"},
    "allow": {"DOES_NOT_ALLOW"}, "allows": {"DOES_NOT_ALLOW"},
    "guarantee": {"GUARANTEES"}, "guarantees": {"GUARANTEES"},
    "protect": {"PROTECTS"}, "protects": {"PROTECTS"},
    "prohibit": {"PROHIBITS"}, "prohibits": {"PROHIBITS"},
    "specify": {"SPECIFIES"}, "specifies": {"SPECIFIES"},
    "adopt": {"ADOPTED"}, "adopted": {"ADOPTED"},
    "draft": {"DRAFTED"}, "drafted": {"DRAFTED"},
    "separation": {"SEPARATED_FROM", "DIVIDED_INTO", "PARTITIONED"},
    "partition": {"PARTITIONED", "DIVIDED_INTO"},
    "separate": {"SEPARATED_FROM", "DIVIDED_INTO"},
    "divided": {"SEPARATED_FROM", "DIVIDED_INTO"},
    "split": {"DIVIDED_INTO"},
    "cause": {"CAUSED_BY"}, "caused": {"CAUSED_BY"},
    "lead": {"LED_TO"}, "leads": {"LED_TO"}, "led": {"LED_TO"},
    "result": {"RESULTED_IN"}, "results": {"RESULTED_IN"},
    "treat": {"TREATS"}, "treats": {"TREATS"}, "used": {"USED_FOR"},
    "use": {"USED_FOR"}, "apply": {"APPLIED_TO"},
    "mention": {"MENTIONS"}, "mentions": {"MENTIONS"},
    "describe": {"DESCRIBES"}, "describes": {"DESCRIBES"},
    "represent": {"REPRESENTS"}, "represents": {"REPRESENTS"},
}

SUMMARY_RELATION_ORDER = [
    "REFLECTS",
    "CELEBRATES",
    "INCLUDES",
    "PRACTICES",
    "WEARS",
    "LOCATED_IN",
    "BASED_IN",
]

MAX_ENTITY_NAME_WORDS = 8
_MULTI_HOP_REGEX = re.compile(
    r"the\s+\w+\s+that\s+\w+|the\s+\w+\s+of\s+the\s+\w+",
    re.IGNORECASE,
)


def _relation_to_text(relation: str) -> str:
    return relation.replace("_", " ").strip().lower()


def _verb_from_relation(relation: str) -> str:
    relation_upper = str(relation or "").strip().replace(" ", "_").upper()
    mapping = {
        "HAS_CAPITAL": "has capital",
        "CAPITAL_OF": "is the capital of",
        "LOCATED_IN": "is located in",
        "BASED_IN": "is based in",
        "INCLUDES": "includes",
        "WEARS": "wears",
        "PRACTICES": "practices",
        "CELEBRATES": "celebrates",
        "FOUNDED": "founded",
        "FOUNDED_BY": "was founded by",
        "DRAFTED": "drafted",
        "CONVENED_IN": "convened in",
        "STARTED": "started",
        "STARTED_IN": "started in",
        "STARTED_ON": "started on",
        "BEGAN": "began",
        "BEGAN_IN": "began in",
        "BEGAN_ON": "began on",
        "CHAIRMAN_OF": "is the chairman of",
        "CEO_OF": "is the CEO of",
        "REFLECTS": "reflects",
        "KNOWN_AS": "is known as",
        "DOES_NOT_ALLOW": "does not allow",
        "GUARANTEES": "guarantees",
        "PROTECTS": "protects",
        "PROHIBITS": "prohibits",
        "SPECIFIES": "specifies",
        "MENTIONS": "mentions",
        "REPRESENTS": "represents",
        "KNOWN_FOR": "is known for",
        "CONDUCTED_BY": "was conducted by",
        "PUBLISHED_IN": "was published in",
        "PUBLISHED_BY": "was published by",
        "TREATS": "treats",
        "CARRIES": "carries",
        "STORES": "stores",
        "BORN_IN": "was born in",
        "DIED_IN": "died in",
        "INVENTED_BY": "was invented by",
        "DISCOVERED_BY": "was discovered by",
        "AUTHORED_BY": "was authored by",
        "WRITTEN_BY": "was written by",
        "LED_BY": "is led by",
        "HEADED_BY": "is headed by",
        "LED_TO": "led to",
        "RESULTED_IN": "resulted in",
        "CAUSED_BY": "was caused by",
        "USED_FOR": "is used for",
        "APPLIED_TO": "is applied to",
        "PART_OF": "is part of",
        "MEMBER_OF": "is a member of",
    }
    if relation_upper in mapping:
        return mapping[relation_upper]
    return _relation_to_text(relation)


def _adjust_verb_for_subject(subject: str, verb: str) -> str:
    lowered_subject = subject.strip().lower()
    last_word = lowered_subject.split()[-1] if lowered_subject.split() else lowered_subject
    if last_word.endswith("s") and not last_word.endswith(("ics", "ss", "us")) and verb.endswith("s"):
        irregular = {
            "is": "are",
            "has": "have",
            "does": "do",
        }
        if verb in irregular:
            return irregular[verb]
        if verb.endswith("ies"):
            return verb[:-3] + "y"
        if verb.endswith(("ches", "shes", "xes", "zes", "sses", "oes")):
            return verb[:-2]
        if verb.endswith("es"):
            return verb[:-1]
        return verb[:-1]
    return verb


def _clean_entity_text(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    if text.isupper():
        if " " not in text:
            return text
        return text.title()
    return text


def _normalize_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def _tokenize_value(value: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(value or "").lower())
        if token and _normalize_token(token)
    ]


def _tokens_match(query_token: str, candidate_token: str) -> bool:
    query_norm = _normalize_token(query_token)
    candidate_norm = _normalize_token(candidate_token)
    if not query_norm or not candidate_norm:
        return False
    if query_norm == candidate_norm:
        return True
    if len(query_norm) >= 5 and len(candidate_norm) >= 5:
        if query_norm in candidate_norm or candidate_norm in query_norm:
            return True
        return SequenceMatcher(None, query_norm, candidate_norm).ratio() >= 0.84
    return False


def _count_term_coverage(text: str, terms: list[str]) -> int:
    tokens = set(_tokenize_value(text))
    covered = 0
    for term in terms:
        if any(_tokens_match(term, token) for token in tokens):
            covered += 1
    return covered


def _required_term_coverage(terms: list[str]) -> int:
    meaningful_terms = [term for term in terms if _normalize_token(term) and _normalize_token(term) not in STOP_WORDS]
    if len(meaningful_terms) <= 2:
        return 1 if meaningful_terms else 0
    return 2


def _entity_name_quality(name: str) -> int:
    tokens = _tokenize_value(name)
    if not tokens:
        return -100
    score = 0
    if len(tokens) <= 2:
        score += 6
    elif len(tokens) <= 4:
        score += 4
    elif len(tokens) <= 6:
        score += 0
    else:
        score -= (len(tokens) - 6) * 4
    if any(len(token) > 16 for token in tokens):
        score -= 2
    return score


def _sentence_from_relation(subject: str, relation: str, object_: str, question: str, terms: list[str]) -> str:
    relation_upper = str(relation or "").strip().replace(" ", "_").upper()
    term_hits_subject = sum(1 for term in terms if term.lower() in subject.lower())
    term_hits_object = sum(1 for term in terms if term.lower() in object_.lower())

    if relation_upper in {"CAPITAL", "HAS_CAPITAL"}:
        return f"The capital of {subject} is {object_}."

    if relation_upper == "CAPITAL_OF":
        return f"The capital of {object_} is {subject}."

    if relation_upper.startswith("HAS_"):
        role = relation_upper[4:].replace("_", " ").lower()
        return f"{object_} is the {role} of {subject}."

    if relation_upper.endswith("_OF"):
        role = relation_upper[:-3].replace("_", " ").lower()
        return f"{subject} is the {role} of {object_}."

    if relation_upper in {"LOCATED_IN", "BASED_IN"}:
        return f"{subject} is located in {object_}."

    if relation_upper == "ADOPTED_ON":
        return f"{subject} was adopted on {object_}."

    if relation_upper in {
        "ADOPTED",
        "FOUNDED",
        "FOUNDED_BY",
        "DRAFTED",
        "CONVENED_IN",
        "CELEBRATES",
        "INCLUDES",
        "PRACTICES",
        "WEARS",
        "REFLECTS",
    }:
        verb = _adjust_verb_for_subject(subject, _verb_from_relation(relation_upper))
        return f"{subject} {verb} {object_}."

    if question.strip().lower().startswith("who is") and term_hits_object > term_hits_subject:
        return f"{object_} is related to {subject} through {relation}."

    if question.strip().lower().startswith("where"):
        return f"{subject} is {relation} {object_}."

    verb = _adjust_verb_for_subject(subject, _verb_from_relation(relation_upper))
    return f"{subject} {verb} {object_}."


def _sentence_from_inverse_relation(anchor: str, subject: str, relation: str) -> str:
    """Render a natural sentence when the anchor is on the object side of a relation."""
    relation_upper = str(relation or "").strip().replace(" ", "_").upper()

    if relation_upper in {"INCLUDES", "CONTAINS", "HAS"}:
        verb = "are included in" if anchor.strip().lower().endswith("s") else "is included in"
        return f"{anchor} {verb} {subject}"

    if relation_upper == "CELEBRATES":
        return f"{anchor} is celebrated by {subject}"

    if relation_upper == "REFLECTS":
        return f"{anchor} is reflected by {subject}"

    if relation_upper == "PRACTICES":
        return f"{anchor} is practiced in {subject}"

    if relation_upper == "WEARS":
        return f"{anchor} is worn in {subject}"

    if relation_upper in {"LOCATED_IN", "BASED_IN"}:
        return f"{subject} is located in {anchor}"

    if relation_upper == "FOUNDED":
        return f"{anchor} was founded by {subject}"

    if relation_upper == "FOUNDED_BY":
        return f"{subject} was founded by {anchor}"

    if relation_upper.startswith("HAS_"):
        role = relation_upper[4:].replace("_", " ").lower()
        return f"{anchor} is the {role} of {subject}"

    if relation_upper.endswith("_OF"):
        role = relation_upper[:-3].replace("_", " ").lower()
        return f"{subject} is the {role} of {anchor}"

    return f"{subject} {_adjust_verb_for_subject(subject, _verb_from_relation(relation_upper))} {anchor}"


def _extract_terms(question: str) -> list[str]:
    """Extract useful search terms without hardcoding any domain entities."""
    if not question.strip():
        return []

    terms: list[str] = []
    seen = set()

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    for left, right in quoted:
        value = (left or right).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            terms.append(value)

    title_case = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    for value in title_case:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in STOP_WORDS and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            terms.append(cleaned)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question):
        lowered = token.lower()
        if lowered in STOP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)

    return terms[:12]


def _extract_entity_phrases(question: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s_-]", " ", question)
    tokens = [token for token in cleaned.split() if token.strip()]
    relation_words = {word.lower() for word in RELATION_HINTS.keys()}
    meaningful = [
        token for token in tokens
        if token.lower() not in STOP_WORDS and token.lower() not in relation_words
    ]
    phrases: list[str] = []

    if len(meaningful) >= 2:
        phrases.append(" ".join(meaningful))

    title_case = re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*)+\b", question)
    for value in title_case:
        phrase = value.strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    return phrases[:4]


def _expand_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen = set()
    for term in terms:
        base = term.strip().lower()
        if not base:
            continue
        candidates = [base]
        if base.endswith("s") and len(base) > 4:
            candidates.append(base[:-1])
        if base.endswith("ing") and len(base) > 5:
            candidates.append(base[:-3])
        candidates.extend(QUESTION_SYNONYMS.get(base, []))

        for candidate in candidates:
            cleaned = candidate.strip().lower()
            if not cleaned or cleaned in seen or cleaned in STOP_WORDS:
                continue
            seen.add(cleaned)
            expanded.append(cleaned)
    return expanded[:20]


def _extract_relation_hints(question: str) -> set[str]:
    hints: set[str] = set()
    capitalized_spans: set[str] = set()
    for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+(?:[a-z]{1,4}|[A-Z][a-zA-Z]+))+)\b", question):
        for word in match.group(0).split():
            if word[:1].isupper():
                capitalized_spans.add(word.lower())

    article_preceded: set[str] = set()
    for match in re.finditer(r"\b(?:the|a|an|this|that)\s+([a-zA-Z]+)\b", question, re.IGNORECASE):
        word = match.group(1).lower()
        always_relation = {"capital", "founder", "father", "member", "leader", "author", "owner", "head"}
        if word not in always_relation:
            article_preceded.add(word)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question.lower()):
        if token in capitalized_spans:
            continue
        if token in article_preceded:
            continue
        hints.update(RELATION_HINTS.get(token, set()))
    return hints


def _extract_relation_groups(question: str) -> list[set[str]]:
    groups: list[set[str]] = []
    seen = set()
    capitalized_spans: set[str] = set()
    for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+(?:[a-z]{1,4}|[A-Z][a-zA-Z]+))+)\b", question):
        for word in match.group(0).split():
            if word[:1].isupper():
                capitalized_spans.add(word.lower())

    article_preceded: set[str] = set()
    always_relation = {"capital", "founder", "father", "member", "leader", "author", "owner", "head"}
    for match in re.finditer(r"\b(?:the|a|an|this|that)\s+([a-zA-Z]+)\b", question, re.IGNORECASE):
        word = match.group(1).lower()
        if word not in always_relation:
            article_preceded.add(word)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question.lower()):
        if token in capitalized_spans or token in article_preceded:
            continue
        relation_group = RELATION_HINTS.get(token)
        if not relation_group:
            continue
        key = tuple(sorted(relation_group))
        if key in seen:
            continue
        seen.add(key)
        groups.append(set(relation_group))
    return groups


def _is_overview_question(question: str) -> bool:
    text = question.strip().lower()
    patterns = [
        "what does this pdf talk about",
        "what is this pdf talking about",
        "what does this document talk about",
        "what is this document talking about",
        "what is this pdf about",
        "what is this document about",
        "give me a summary",
        "summarize",
        "summary",
        "main topics",
        "key topics",
        "overview",
    ]
    return any(pattern in text for pattern in patterns)


def _is_node_overview_question(question: str) -> bool:
    text = question.strip().lower()
    return any(pattern in text for pattern in GENERIC_NODE_PATTERNS)


def _is_plain_entity_query(question: str) -> bool:
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question.lower()) if token not in STOP_WORDS]
    return bool(tokens) and len(tokens) <= 4 and "?" not in question


def _requires_phrase_match(question: str, relation_hints: set[str], phrases: list[str]) -> bool:
    if relation_hints or not phrases:
        return False
    text = question.strip().lower()
    if text.startswith(("what is ", "what are ", "who is ", "who was ")):
        return True
    if _is_node_overview_question(question):
        return True
    if text.startswith(("what ", "who ", "when ", "where ", "which ", "how ")):
        return False
    if _is_plain_entity_query(question):
        return True
    return False


def _is_explicit_relation_question(question: str, relation_hints: set[str]) -> bool:
    """Avoid treating nouns inside topic phrases as relation requests."""
    if not relation_hints:
        return False

    text = question.strip().lower()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question)
    relation_word_present = any(token.lower() in RELATION_HINTS for token in tokens)
    has_capitalized_phrase = bool(
        re.search(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)+\b", question)
    )
    if text.startswith(("what does ", "what did ", "when ", "where ", "who ", "which ")):
        return True
    if re.match(r"what\s+(is|are)\s+the\s+", text):
        return True
    if _is_node_overview_question(question):
        return False
    if _is_plain_entity_query(question) and relation_word_present and has_capitalized_phrase:
        return True
    if _is_plain_entity_query(question) and not text.startswith(("what is the ", "what are the ", "where ", "when ", "who ")):
        return False
    return True


def _is_multi_hop_question(question: str, relation_groups: list[set[str]]) -> bool:
    """Detect questions that likely need a path chain instead of one local edge."""
    text = question.strip().lower()
    if len(relation_groups) >= 2:
        return True
    chain_markers = [
        "adopted on",
        "adopted by",
        "founded by",
        "based in",
        "located in",
        "connected to",
        "related to",
        "who founded",
        "who created",
        "who invented",
        "who discovered",
        "who wrote",
        "who authored",
        "who led",
        "who started",
        "where is the",
        "where was the",
        "where did the",
        "what country",
        "which country",
        "what city",
        "which city",
        "what year",
        "which year",
        "when did",
        "when was",
        "what caused",
        "what led to",
        "what resulted in",
        "who is the leader",
        "who is the head",
        "who is the founder",
        "capital of",
        "headquarters of",
        "founder of",
        "member of",
        "part of",
        "author of",
        "owner of",
    ]
    if any(marker in text for marker in chain_markers):
        return True
    if _MULTI_HOP_REGEX.search(question):
        return True
    return False


def _score_direct_row(row: dict, terms: list[str]) -> int:
    subject = _clean_entity_text(row.get("from_name", "")).lower()
    object_ = _clean_entity_text(row.get("to_name", "")).lower()
    relation = _relation_to_text(str(row.get("relation_type", "")).strip())
    searchable_text = str(row.get("searchable_text", "")).lower()
    subject_tokens = set(_tokenize_value(subject))
    object_tokens = set(_tokenize_value(object_))
    relation_tokens = set(_tokenize_value(relation))
    score = 0

    for term in terms:
        token = _normalize_token(term)
        if any(_tokens_match(token, item) for item in subject_tokens):
            score += 3
        elif token and token in subject:
            score += 1
        if any(_tokens_match(token, item) for item in object_tokens):
            score += 3
        elif token and token in object_:
            score += 1
        if any(_tokens_match(token, item) for item in relation_tokens):
            score += 2
        elif token and token in relation:
            score += 1
        if token and token in searchable_text:
            score += 1

    if subject and object_:
        score += 1
    score += _entity_name_quality(subject)
    score += _entity_name_quality(object_)
    return score


def _format_direct_results(question: str, results: list[dict], terms: list[str], relation_hints: set[str], phrases: list[str] | None = None) -> str:
    cleaned_rows = []
    for row in results:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = _relation_to_text(str(row.get("relation_type", "")).strip())
        object_ = _clean_entity_text(row.get("to_name", ""))
        if not subject or not relation or not object_:
            continue
        cleaned_rows.append(
            {
                "subject": subject,
                "relation": relation,
                "object": object_,
                "score": _score_direct_row(row, terms),
            }
        )

    if not cleaned_rows:
        return ""

    if _requires_phrase_match(question, relation_hints, phrases or []):
        phrase_rows = [
            row for row in cleaned_rows
            if _phrase_coverage(f"{row['subject']} {row['object']}", phrases or []) > 0
        ]
        if not phrase_rows:
            return ""
        cleaned_rows = phrase_rows

    minimum_coverage = _required_term_coverage(terms)
    if minimum_coverage >= 2:
        covered_rows = [
            row for row in cleaned_rows
            if _count_term_coverage(f"{row['subject']} {row['relation']} {row['object']}", terms) >= minimum_coverage
        ]
        if covered_rows:
            cleaned_rows = covered_rows
        elif relation_hints:
            return ""

    if relation_hints:
        grouped: dict[tuple[str, str], list[str]] = {}
        for row in sorted(cleaned_rows, key=lambda item: item["score"], reverse=True):
            relation_upper = row["relation"].replace(" ", "_").upper()
            if relation_upper not in relation_hints:
                continue
            key = (row["subject"], relation_upper)
            grouped.setdefault(key, [])
            if row["object"] not in grouped[key]:
                grouped[key].append(row["object"])

        if grouped:
            best_key = max(grouped.keys(), key=lambda key: len(grouped[key]))
            subject, relation_upper = best_key
            values = grouped[best_key]
            if relation_upper in {"HAS_CAPITAL", "CAPITAL"} and values:
                return f"The capital of {subject} is {values[0]}."
            if relation_upper == "CAPITAL_OF" and values:
                return f"The capital of {values[0]} is {subject}."
            if relation_upper == "CELEBRATES":
                return f"{subject} celebrates {_join_values(values)}."
            if relation_upper == "INCLUDES":
                return f"{subject} includes {_join_values(values)}."
            if relation_upper in {"LOCATED_IN", "BASED_IN"} and values:
                return f"{subject} is located in {values[0]}."
            verb = _adjust_verb_for_subject(subject, _verb_from_relation(relation_upper))
            return f"{subject} {verb} {_join_values(values)}."
        return ""

    lower_question = question.strip().lower()
    if not relation_hints and (_is_plain_entity_query(question) or lower_question.startswith("what is") or lower_question.startswith("what are")):
        grouped_by_subject: dict[str, dict[str, list[str]]] = {}
        for row in sorted(cleaned_rows, key=lambda item: item["score"], reverse=True):
            subject = row["subject"]
            relation_upper = row["relation"].replace(" ", "_").upper()
            grouped_by_subject.setdefault(subject, {})
            grouped_by_subject[subject].setdefault(relation_upper, [])
            if row["object"] not in grouped_by_subject[subject][relation_upper]:
                grouped_by_subject[subject][relation_upper].append(row["object"])

        best_subject = max(
            grouped_by_subject.keys(),
            key=lambda subject: (
                sum(len(values) for values in grouped_by_subject[subject].values()),
                sum(1 for term in terms if term.lower() in subject.lower()),
            ),
        )

        ordered_relations = [rel for rel in SUMMARY_RELATION_ORDER if rel in grouped_by_subject[best_subject]]
        ordered_relations.extend(
            rel for rel in grouped_by_subject[best_subject].keys()
            if rel not in ordered_relations
        )
        lines = []
        for relation_upper in ordered_relations:
            values = grouped_by_subject[best_subject][relation_upper]
            verb = _adjust_verb_for_subject(best_subject, _verb_from_relation(relation_upper))
            lines.append(f"{best_subject} {verb} {_join_values(values)}")
        if lines:
            if len(lines) == 1:
                return lines[0] + "."
            return "Based on the uploaded document: " + "; ".join(lines) + "."

    best = sorted(
        cleaned_rows,
        key=lambda row: (row["score"], len(row["subject"]) + len(row["object"])),
        reverse=True,
    )[0]

    if lower_question.startswith("who is"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    if lower_question.startswith("what is"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    if lower_question.startswith("where"):
        return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)
    return _sentence_from_relation(best["subject"], best["relation"], best["object"], question, terms)


def _score_path_row(row: dict, terms: list[str], relation_hints: set[str]) -> int:
    nodes = [_clean_entity_text(name).lower() for name in row.get("path_nodes", []) if _clean_entity_text(name)]
    relations = [str(rel).strip().replace(" ", "_").upper() for rel in row.get("path_relationships", []) if str(rel).strip()]
    score = 0
    for relation in relations:
        if relation in relation_hints:
            score += 8
    for term in terms:
        token = term.lower()
        for node in nodes:
            if token in node:
                score += 2
        for relation in relations:
            if token in _relation_to_text(relation).lower():
                score += 2
    score += len(relations)
    return score


def _format_path_results(results: list[dict], terms: list[str], relation_hints: set[str]) -> str:
    ranked_rows = sorted(
        results,
        key=lambda row: _score_path_row(row, terms, relation_hints),
        reverse=True,
    )
    lines: list[str] = []
    seen = set()

    for row in ranked_rows:
        nodes = [_clean_entity_text(name) for name in row.get("path_nodes", []) if _clean_entity_text(name)]
        relations = [str(rel).strip() for rel in row.get("path_relationships", []) if str(rel).strip()]
        if len(nodes) < 2 or not relations:
            continue

        parts = []
        for idx, relation in enumerate(relations):
            if idx + 1 >= len(nodes):
                break
            subject = nodes[idx]
            object_ = nodes[idx + 1]
            phrase = _sentence_from_relation(subject, relation, object_, "", [])
            parts.append(phrase.rstrip("."))

        line = ", and ".join(parts)
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= 3:
            break

    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0] + "."
    return "Based on the uploaded document: " + "; ".join(lines) + "."


def _format_overview(results: list[dict]) -> str:
    topics: list[str] = []
    seen = set()

    for row in results:
        subject = _clean_entity_text(row.get("from_name", ""))
        object_ = _clean_entity_text(row.get("to_name", ""))
        for item in (subject, object_):
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                topics.append(item)
            if len(topics) >= 10:
                break
        if len(topics) >= 10:
            break

    if not topics:
        return ""
    if len(topics) == 1:
        return f"The uploaded document mainly discusses {topics[0]}."
    return "The uploaded document mainly discusses " + ", ".join(topics[:-1]) + f", and {topics[-1]}."


def _join_values(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _score_text_passage(text: str, terms: list[str], phrases: list[str], relation_hints: set[str], question: str) -> int:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return -10

    tokens = set(_tokenize_value(lowered))
    score = 0
    coverage = _count_term_coverage(lowered, terms)
    score += coverage * 8

    for phrase in phrases:
        phrase_lower = phrase.lower().strip()
        phrase_tokens = [token for token in _tokenize_value(phrase_lower) if token not in STOP_WORDS]
        if phrase_lower and phrase_lower in lowered:
            score += 14
        elif phrase_tokens and all(
            any(_tokens_match(token, candidate) for candidate in tokens)
            for token in phrase_tokens
        ):
            score += 10 + len(phrase_tokens)

    for term in terms:
        token = _normalize_token(term)
        if any(_tokens_match(token, candidate) for candidate in tokens):
            score += 3

    for relation in relation_hints:
        relation_text = _relation_to_text(relation)
        relation_tokens = [token for token in _tokenize_value(relation_text) if token not in STOP_WORDS]
        if relation_text and relation_text in lowered:
            score += 5
        elif relation_tokens and all(
            any(_tokens_match(token, candidate) for candidate in tokens)
            for token in relation_tokens
        ):
            score += 4

    if question.strip().lower().startswith("when ") and re.search(r"\b\d{4}\b|\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", text):
        score += 5
    if question.strip().lower().startswith("who ") and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        score += 4
    return score


def _normalise_score(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / cap))


def _graph_entity_boost(searchable_text: str, graph_entities: list[str]) -> float:
    if not graph_entities:
        return 0.0
    lowered = searchable_text.lower()
    hits = 0
    for entity in graph_entities:
        entity_text = _clean_entity_text(entity).lower()
        if entity_text and entity_text in lowered:
            hits += 1
    return min(1.0, hits / max(1, min(len(graph_entities), 5)))


def _hybrid_chunk_score(
    *,
    embedding_score: float,
    keyword_score: float,
    fuzzy_score: float,
    graph_score: float,
) -> float:
    return (
        HYBRID_WEIGHTS["embedding"] * embedding_score
        + HYBRID_WEIGHTS["keyword"] * keyword_score
        + HYBRID_WEIGHTS["fuzzy"] * fuzzy_score
        + HYBRID_WEIGHTS["graph"] * graph_score
    )


def _search_semantic_chunks(
    document_id: str,
    question: str,
    terms: list[str],
    phrases: list[str],
    relation_hints: set[str],
    graph_entities: list[str] | None = None,
    limit: int = 5,
) -> list[dict]:
    chunks = get_document_chunks(document_id)
    if not chunks:
        return []

    hits: list[dict] = []
    query_embedding = embed_text(question)
    graph_entities = graph_entities or []

    for chunk in chunks:
        chunk_text = str(chunk.get("text", "") if isinstance(chunk, dict) else chunk).strip()
        if not chunk_text:
            continue
        keywords = chunk.get("keywords", []) if isinstance(chunk, dict) else []
        searchable_text = " ".join(
            [
                chunk_text,
                str(chunk.get("summary", "") if isinstance(chunk, dict) else ""),
                " ".join(keywords if isinstance(keywords, list) else []),
                str(chunk.get("section", "") if isinstance(chunk, dict) else ""),
            ]
        )
        fuzzy_raw = _score_text_passage(searchable_text, terms, phrases, relation_hints, question)
        keyword_raw = _count_term_coverage(" ".join(keywords if isinstance(keywords, list) else []), terms)
        if not keyword_raw:
            keyword_raw = _count_term_coverage(str(chunk.get("summary", "") if isinstance(chunk, dict) else ""), terms)

        embedding_score = cosine_similarity(query_embedding, chunk.get("embedding", []) if isinstance(chunk, dict) else [])
        keyword_score = _normalise_score(keyword_raw, max(1, min(len(terms), 5)))
        fuzzy_score = _normalise_score(fuzzy_raw, 40)
        graph_score = _graph_entity_boost(searchable_text, graph_entities)
        final_score = _hybrid_chunk_score(
            embedding_score=embedding_score,
            keyword_score=keyword_score,
            fuzzy_score=fuzzy_score,
            graph_score=graph_score,
        )

        if final_score < MIN_HYBRID_SCORE:
            continue

        candidate_sentences = split_into_sentences(chunk_text) or [chunk_text]
        ranked_sentences = sorted(
            candidate_sentences,
            key=lambda sentence: _score_text_passage(sentence, terms, phrases, relation_hints, question),
            reverse=True,
        )

        selected_sentences: list[str] = []
        seen = set()
        for sentence in ranked_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_key = sentence.lower()
            sentence_score = _score_text_passage(sentence, terms, phrases, relation_hints, question)
            if sentence_score <= 0 and selected_sentences:
                continue
            if sentence_key in seen:
                continue
            seen.add(sentence_key)
            selected_sentences.append(sentence)
            if len(selected_sentences) >= 3:
                break

        if not selected_sentences:
            selected_sentences = [chunk.strip()]

        hits.append(
            {
                "chunk": chunk,
                "chunk_id": chunk.get("id", "") if isinstance(chunk, dict) else "",
                "text": chunk_text,
                "score": final_score,
                "embedding_score": embedding_score,
                "keyword_score": keyword_score,
                "fuzzy_score": fuzzy_score,
                "graph_score": graph_score,
                "sentences": selected_sentences,
                "summary": chunk.get("summary", "") if isinstance(chunk, dict) else "",
                "keywords": chunk.get("keywords", []) if isinstance(chunk, dict) else [],
                "section": chunk.get("section", "Document") if isinstance(chunk, dict) else "Document",
                "page": chunk.get("page") if isinstance(chunk, dict) else None,
            }
        )

    reranked = sorted(
        sorted(hits, key=lambda hit: hit["score"], reverse=True)[:15],
        key=lambda hit: (
            hit["score"],
            _count_term_coverage(" ".join(hit.get("sentences", [])), terms),
            hit.get("graph_score", 0),
        ),
        reverse=True,
    )
    return reranked[:limit]


def _format_chunk_answer(
    question: str,
    hits: list[dict],
    terms: list[str],
    phrases: list[str] | None = None,
    relation_hints: set[str] | None = None,
) -> str:
    if not hits:
        return ""

    phrases = phrases or []
    relation_hints = relation_hints or set()
    chosen_sentences: list[str] = []
    seen = set()
    for hit in hits:
        for sentence in hit.get("sentences", []):
            sentence = sentence.strip()
            if not sentence:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen_sentences.append(sentence)
            if len(chosen_sentences) >= 3:
                break
        if len(chosen_sentences) >= 3:
            break

    if not chosen_sentences:
        return ""

    all_sentences = list(chosen_sentences)
    lower_question = question.strip().lower()
    preferred_sentences: list[str] = []
    if lower_question.startswith("where ") and relation_hints & {"LOCATED_IN", "BASED_IN"}:
        preferred_sentences = [
            sentence for sentence in all_sentences
            if "located in" in sentence.lower() or "based in" in sentence.lower()
        ]
    elif lower_question.startswith("when "):
        preferred_sentences = [
            sentence for sentence in all_sentences
            if re.search(r"\b\d{4}\b|\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", sentence)
        ]
    if preferred_sentences:
        chosen_sentences = preferred_sentences

    minimum_coverage = _required_term_coverage(terms)
    filtered = [
        sentence for sentence in chosen_sentences
        if _count_term_coverage(sentence, terms) >= minimum_coverage
    ]
    if filtered:
        chosen_sentences = filtered

    if _requires_phrase_match(question, relation_hints, phrases):
        phrase_filtered = [
            sentence for sentence in chosen_sentences
            if _phrase_coverage(sentence, phrases) > 0
        ]
        if not phrase_filtered:
            return ""
        chosen_sentences = phrase_filtered

    if preferred_sentences:
        preferred_remaining = [sentence for sentence in chosen_sentences if sentence in preferred_sentences]
        if preferred_remaining:
            chosen_sentences = preferred_remaining
        else:
            chosen_sentences = preferred_sentences

    if lower_question.startswith(("when ", "where ", "who ")):
        chosen_sentences = chosen_sentences[:1]

    if len(chosen_sentences) == 1:
        sentence = chosen_sentences[0]
        if lower_question.startswith(("when ", "where ", "who ")):
            sentence = re.sub(r"^The\s+(?=[A-Z])", "", sentence)
        return sentence if sentence.endswith((".", "!", "?")) else sentence + "."
    return "Based on the uploaded document: " + " ".join(
        sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        for sentence in chosen_sentences
    )


def _answer_term_coverage(answer: str, terms: list[str]) -> int:
    return _count_term_coverage(answer, terms)


def _phrase_coverage(text: str, phrases: list[str]) -> int:
    if not text.strip() or not phrases:
        return 0
    tokens = set(_tokenize_value(text))
    coverage = 0
    for phrase in phrases:
        phrase_tokens = [token for token in _tokenize_value(phrase) if token not in STOP_WORDS]
        if phrase_tokens and all(
            any(_tokens_match(token, candidate) for candidate in tokens)
            for token in phrase_tokens
        ):
            coverage += 1
    return coverage


def _prefer_chunk_answer(question: str, graph_answer: str, chunk_answer: str, terms: list[str], relation_hints: set[str], phrases: list[str]) -> bool:
    if not chunk_answer:
        return False
    if not graph_answer:
        return True

    graph_coverage = _answer_term_coverage(graph_answer, terms)
    chunk_coverage = _answer_term_coverage(chunk_answer, terms)
    minimum_coverage = _required_term_coverage(terms)
    graph_phrase_coverage = _phrase_coverage(graph_answer, phrases)
    chunk_phrase_coverage = _phrase_coverage(chunk_answer, phrases)

    if _is_overview_question(question):
        return False

    if relation_hints:
        return graph_coverage < minimum_coverage and chunk_coverage >= minimum_coverage

    if chunk_coverage >= minimum_coverage and graph_coverage < minimum_coverage:
        return True

    if chunk_phrase_coverage > graph_phrase_coverage and chunk_coverage >= graph_coverage:
        return True

    if len(graph_answer.split()) > 35 and graph_coverage == 0 and chunk_coverage > 0:
        return True

    return False


def _response_with_chunk_fallback(base_response: dict, question: str, chunk_hits: list[dict], terms: list[str], relation_hints: set[str], phrases: list[str]) -> dict:
    chunk_answer = _format_chunk_answer(
        question,
        chunk_hits,
        terms,
        phrases=phrases,
        relation_hints=relation_hints,
    )
    if not _prefer_chunk_answer(question, base_response.get("answer", ""), chunk_answer, terms, relation_hints, phrases):
        return base_response

    steps = list(base_response.get("steps_taken", []))
    steps.append(
        {
            "step": "semantic_chunk_lookup",
            "matches": len(chunk_hits),
            "terms": terms,
        }
    )
    return {
        **base_response,
        "query": "HYBRID_GRAPH_PLUS_CHUNKS",
        "results": base_response.get("results", []) + [
            {
                "chunk": hit["chunk"],
                "chunk_id": hit.get("chunk_id", ""),
                "sentences": hit["sentences"],
                "score": hit["score"],
                "score_breakdown": {
                    "embedding": hit.get("embedding_score", 0),
                    "keyword": hit.get("keyword_score", 0),
                    "fuzzy": hit.get("fuzzy_score", 0),
                    "graph": hit.get("graph_score", 0),
                },
                "summary": hit.get("summary", ""),
                "keywords": hit.get("keywords", []),
                "section": hit.get("section", "Document"),
                "page": hit.get("page"),
            }
            for hit in chunk_hits[:3]
        ],
        "answer": chunk_answer,
        "steps_taken": steps,
    }


def _rank_anchor_entities(terms: list[str], phrases: list[str], candidates: list[dict]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        name = _clean_entity_text(candidate.get("entity_name", ""))
        lowered = name.lower()
        normalized_name = _normalize_token(name)
        name_tokens = set(_tokenize_value(name))
        score = 0
        has_match = False
        for phrase in phrases:
            phrase_lower = phrase.lower()
            normalized_phrase = _normalize_token(phrase)
            phrase_tokens = [token for token in _tokenize_value(phrase) if token not in STOP_WORDS]
            if phrase_lower == lowered:
                score += 16
                has_match = True
            elif normalized_phrase == normalized_name:
                score += 14
                has_match = True
            elif phrase_tokens and all(token in name_tokens for token in phrase_tokens):
                score += 12 + len(phrase_tokens)
                has_match = True
            elif phrase_tokens and all(
                any(_tokens_match(token, name_token) for name_token in name_tokens)
                for token in phrase_tokens
            ):
                score += 10 + len(phrase_tokens)
                has_match = True
            elif phrase_tokens and len(phrase_tokens) > 1 and not all(token in name_tokens for token in phrase_tokens):
                score -= 4
            elif phrase_lower in lowered:
                score += 6
                has_match = True
        for term in terms:
            token = _normalize_token(term)
            if token == lowered:
                score += 6
                has_match = True
            elif _normalize_token(token) == normalized_name:
                score += 5
                has_match = True
            elif any(_tokens_match(token, name_token) for name_token in name_tokens):
                score += 4
                has_match = True
            elif token and len(token) >= 5 and token in lowered:
                score += 3
                has_match = True
        score += _entity_name_quality(name)
        if has_match:
            scored.append((score, name))

    ranked = sorted(scored, key=lambda item: (item[0], -len(item[1])), reverse=True)
    ordered_names: list[str] = []
    seen = set()
    for _, name in ranked:
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered_names.append(name)
    return ordered_names


def _score_neighborhood_row(row: dict, anchor: str, terms: list[str], relation_hints: set[str]) -> int:
    subject = _clean_entity_text(row.get("from_name", ""))
    object_ = _clean_entity_text(row.get("to_name", ""))
    relation = str(row.get("relation_type", "")).strip().upper()
    score = 0

    if subject.lower() == anchor.lower() or object_.lower() == anchor.lower():
        score += 5
    if relation in relation_hints:
        score += 8

    subject_lower = subject.lower()
    object_lower = object_.lower()
    relation_text = _relation_to_text(relation).lower()
    subject_tokens = set(_tokenize_value(subject))
    object_tokens = set(_tokenize_value(object_))
    relation_tokens = set(_tokenize_value(relation_text))
    for term in terms:
        token = _normalize_token(term)
        if any(_tokens_match(token, item) for item in subject_tokens):
            score += 2
        elif token and len(token) >= 5 and token in subject_lower:
            score += 1
        if any(_tokens_match(token, item) for item in object_tokens):
            score += 2
        elif token and len(token) >= 5 and token in object_lower:
            score += 1
        if any(_tokens_match(token, item) for item in relation_tokens):
            score += 2
        elif token and len(token) >= 5 and token in relation_text:
            score += 1
    score += _entity_name_quality(subject)
    score += _entity_name_quality(object_)
    return score


def _format_entity_neighborhood(question: str, anchor: str, rows: list[dict], terms: list[str], relation_hints: set[str], relation_groups: list[set[str]]) -> str:
    facts: list[str] = []
    seen = set()
    lowered_anchor = anchor.lower()
    grouped_objects: dict[str, list[str]] = {}
    incoming_relations: dict[str, list[str]] = {}
    incoming_facts: list[str] = []

    ranked_rows = sorted(
        rows,
        key=lambda row: _score_neighborhood_row(row, anchor, terms, relation_hints),
        reverse=True,
    )

    for row in ranked_rows:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = str(row.get("relation_type", "")).strip()
        object_ = _clean_entity_text(row.get("to_name", ""))
        relation_upper = relation.replace(" ", "_").upper()
        if not subject or not relation or not object_:
            continue

        if subject.lower() == lowered_anchor:
            grouped_objects.setdefault(relation_upper, [])
            if object_ not in grouped_objects[relation_upper]:
                grouped_objects[relation_upper].append(object_)
        elif object_.lower() == lowered_anchor:
            incoming_relations.setdefault(relation_upper, [])
            if subject not in incoming_relations[relation_upper]:
                incoming_relations[relation_upper].append(subject)
            line = _sentence_from_inverse_relation(anchor, subject, relation).rstrip(".")
            if line and line.lower() not in seen:
                seen.add(line.lower())
                incoming_facts.append(line)

    lower_question = question.strip().lower()
    if relation_groups:
        available_relations = set(grouped_objects.keys()) | set(incoming_relations.keys())
        if not all(group & available_relations for group in relation_groups):
            return ""

    if relation_hints & {"HAS_CAPITAL", "CAPITAL", "CAPITAL_OF"}:
        if "HAS_CAPITAL" in grouped_objects:
            value = grouped_objects["HAS_CAPITAL"][0]
            return f"The capital of {anchor} is {value}."
        if "CAPITAL" in grouped_objects:
            value = grouped_objects["CAPITAL"][0]
            return f"The capital of {anchor} is {value}."
        for row in ranked_rows:
            relation_upper = str(row.get("relation_type", "")).strip().replace(" ", "_").upper()
            subject = _clean_entity_text(row.get("from_name", ""))
            object_ = _clean_entity_text(row.get("to_name", ""))
            if relation_upper == "CAPITAL_OF" and object_.lower() == lowered_anchor:
                return f"The capital of {anchor} is {subject}."

    if relation_hints:
        if "CELEBRATES" in relation_hints and "CELEBRATES" in grouped_objects:
            values = grouped_objects["CELEBRATES"]
            return f"{anchor} celebrates {_join_values(values)}."
        if "INCLUDES" in relation_hints and "INCLUDES" in grouped_objects:
            values = grouped_objects["INCLUDES"]
            return f"{anchor} includes {_join_values(values)}."
        if relation_hints & {"LOCATED_IN", "BASED_IN"}:
            for relation_key in ("LOCATED_IN", "BASED_IN"):
                if relation_key in grouped_objects and grouped_objects[relation_key]:
                    return f"{anchor} is located in {grouped_objects[relation_key][0]}."
        if relation_hints & {"PRACTICES"} and "PRACTICES" in grouped_objects:
            return f"{anchor} practices {_join_values(grouped_objects['PRACTICES'])}."
        if relation_hints & {"WEARS"} and "WEARS" in grouped_objects:
            return f"{anchor} wears {_join_values(grouped_objects['WEARS'])}."
        if relation_hints & {"FOUNDED", "FOUNDED_BY"}:
            if "FOUNDED" in grouped_objects:
                return f"{anchor} founded {_join_values(grouped_objects['FOUNDED'])}."
            if "FOUNDED_BY" in grouped_objects:
                return f"{anchor} was founded by {_join_values(grouped_objects['FOUNDED_BY'])}."
        if relation_hints & {"STARTED", "STARTED_IN", "STARTED_ON", "BEGAN", "BEGAN_IN", "BEGAN_ON", "CONVENED_IN"}:
            for relation_upper in ("STARTED_ON", "STARTED_IN", "STARTED", "BEGAN_ON", "BEGAN_IN", "BEGAN", "CONVENED_IN"):
                if relation_upper in grouped_objects and grouped_objects[relation_upper]:
                    values = grouped_objects[relation_upper]
                    verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation_upper))
                    return f"{anchor} {verb} {_join_values(values)}."
        matched_lines = []
        for relation_upper in sorted(relation_hints):
            if relation_upper not in grouped_objects:
                continue
            values = grouped_objects[relation_upper]
            verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation_upper))
            matched_lines.append(f"{anchor} {verb} {_join_values(values)}")
        if matched_lines:
            return "; ".join(matched_lines) + "."
        for relation_upper in sorted(relation_hints):
            if relation_upper in incoming_relations:
                return "; ".join(
                    _sentence_from_inverse_relation(anchor, subject, relation_upper).rstrip(".")
                    for subject in incoming_relations[relation_upper]
                ) + "."
        return ""

    if lower_question.startswith("what does") and "INCLUDES" in grouped_objects:
        values = grouped_objects["INCLUDES"]
        if len(values) == 1:
            return f"{anchor} includes {values[0]}."
        return f"{anchor} includes {_join_values(values)}."

    if not relation_hints and (
        lower_question.startswith("what is")
        or lower_question.startswith("what are")
        or _is_node_overview_question(question)
        or _is_plain_entity_query(question)
    ):
        grouped_by_verb: dict[str, list[str]] = {}
        ordered_relations = [rel for rel in SUMMARY_RELATION_ORDER if rel in grouped_objects]
        ordered_relations.extend(rel for rel in grouped_objects.keys() if rel not in ordered_relations)
        for relation_upper in ordered_relations:
            values = grouped_objects.get(relation_upper, [])
            if not values:
                continue
            verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation_upper))
            grouped_by_verb.setdefault(verb, [])
            for value in values:
                if value not in grouped_by_verb[verb]:
                    grouped_by_verb[verb].append(value)
        summary_lines = [f"{anchor} {verb} {_join_values(values)}" for verb, values in grouped_by_verb.items() if values]
        summary_lines.extend(line for line in incoming_facts if line not in summary_lines)
        if summary_lines:
            if len(summary_lines) == 1:
                return summary_lines[0] + "."
            return "Based on the uploaded document: " + "; ".join(summary_lines) + "."

    for row in ranked_rows:
        subject = _clean_entity_text(row.get("from_name", ""))
        relation = str(row.get("relation_type", "")).strip()
        object_ = _clean_entity_text(row.get("to_name", ""))
        if not subject or not relation or not object_:
            continue

        if subject.lower() == lowered_anchor:
            verb = _adjust_verb_for_subject(anchor, _verb_from_relation(relation))
            line = f"{anchor} {verb} {object_}"
        elif object_.lower() == lowered_anchor:
            line = _sentence_from_inverse_relation(anchor, subject, relation)
        else:
            continue

        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(line)
        if len(facts) >= 8:
            break

    if not facts:
        return ""

    if question.strip().lower().startswith("what are"):
        return facts[0] + "."
    if question.strip().lower().startswith("who is") or question.strip().lower().startswith("what is"):
        return facts[0] + "."
    return "Based on the uploaded document: " + "; ".join(facts) + "."


def ask_question(question: str, document_id: str) -> dict:
    """Graph-first question answering scoped to the active uploaded document."""
    graph = GraphStore()
    try:
        total_triples = graph.count_triples(document_id)
        terms = _extract_terms(question)
        entity_phrases = _extract_entity_phrases(question)
        expanded_terms = _expand_terms(terms)
        relation_hints = _extract_relation_hints(question)
        relation_groups = _extract_relation_groups(question)
        if not _is_explicit_relation_question(question, relation_hints):
            relation_hints = set()
            relation_groups = []
        should_use_multi_hop = _is_multi_hop_question(question, relation_groups)
        should_prioritize_path = len(relation_groups) >= 2 or _MULTI_HOP_REGEX.search(question) is not None
        entity_search_terms = entity_phrases + (terms or expanded_terms)
        entity_cypher, entity_candidates = graph.find_relevant_entities(entity_search_terms, document_id, limit=12)
        anchor_entities = _rank_anchor_entities(terms or expanded_terms, entity_phrases, entity_candidates)
        if _requires_phrase_match(question, relation_hints, entity_phrases):
            anchor_entities = [
                entity_name
                for entity_name in anchor_entities
                if _phrase_coverage(entity_name, entity_phrases) > 0
            ]
        graph_expanded_terms = list(expanded_terms)
        for entity_name in anchor_entities[:5]:
            for token in _tokenize_value(entity_name):
                if token not in graph_expanded_terms and token not in STOP_WORDS:
                    graph_expanded_terms.append(token)
        chunk_hits = _search_semantic_chunks(
            document_id,
            question,
            graph_expanded_terms,
            entity_phrases,
            relation_hints,
            graph_entities=anchor_entities[:8],
            limit=5,
        )

        if _is_overview_question(question):
            summary = graph.get_document_summary(document_id)
            if summary:
                response = {
                    "triples_added": total_triples,
                    "query": "DOCUMENT_SUMMARY_LOOKUP",
                    "results": [{"summary": summary}],
                    "answer": summary,
                    "steps_taken": [
                        {
                            "step": "document_summary",
                            "terms": graph_expanded_terms,
                            "matches": 1,
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

            overview_cypher, overview_results = graph.get_graph_overview(document_id, limit=12)
            if overview_results:
                response = {
                    "triples_added": total_triples,
                    "query": overview_cypher,
                    "results": overview_results,
                    "answer": _format_overview(overview_results),
                    "steps_taken": [
                        {
                            "step": "graph_overview",
                            "terms": graph_expanded_terms,
                            "matches": len(overview_results),
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        path_cypher = ""
        if should_prioritize_path:
            path_cypher, path_results = graph.search_paths(expanded_terms, document_id, max_hops=3, limit=80)
            path_answer = _format_path_results(path_results, expanded_terms, relation_hints) if path_results else ""
            if path_answer:
                response = {
                    "triples_added": total_triples,
                    "query": path_cypher,
                    "results": path_results,
                    "answer": path_answer,
                    "steps_taken": [
                        {
                            "step": "multi_hop_lookup",
                            "terms": graph_expanded_terms,
                            "relation_hints": sorted(relation_hints),
                            "matches": len(path_results),
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        for anchor_entity in anchor_entities[:5]:
            if not relation_hints and _required_term_coverage(expanded_terms) >= 2:
                if _count_term_coverage(anchor_entity, expanded_terms) < _required_term_coverage(expanded_terms):
                    continue
            neighborhood_cypher, neighborhood_rows = graph.get_entity_neighborhood(anchor_entity, document_id, limit=200)
            if not neighborhood_rows:
                continue

            neighborhood_answer = _format_entity_neighborhood(
                question,
                anchor_entity,
                neighborhood_rows,
                expanded_terms,
                relation_hints,
                relation_groups,
            )
            if not neighborhood_answer:
                continue

            step_name = "entity_neighborhood"
            if _is_node_overview_question(question):
                step_name = "entity_overview"

            response = {
                "triples_added": total_triples,
                "query": neighborhood_cypher,
                "results": neighborhood_rows,
                "answer": neighborhood_answer,
                "steps_taken": [
                    {
                        "step": step_name,
                        "anchor": anchor_entity,
                        "terms": graph_expanded_terms,
                        "relation_hints": sorted(relation_hints),
                        "matches": len(neighborhood_rows),
                    }
                ],
            }
            return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        direct_cypher, direct_results = graph.search_related(expanded_terms, document_id, limit=25)
        if direct_results and len(relation_groups) < 2:
            direct_answer = _format_direct_results(
                question,
                direct_results,
                expanded_terms,
                relation_hints,
                phrases=entity_phrases,
            )
            if direct_answer:
                response = {
                    "triples_added": total_triples,
                    "query": direct_cypher,
                    "results": direct_results,
                    "answer": direct_answer,
                    "steps_taken": [
                        {
                            "step": "one_hop_lookup",
                            "terms": expanded_terms,
                            "matches": len(direct_results),
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        semantic_cypher, semantic_results = graph.search_semantic(expanded_terms, document_id, limit=40)
        if semantic_results and len(relation_groups) < 2:
            semantic_answer = _format_direct_results(
                question,
                semantic_results,
                expanded_terms,
                relation_hints,
                phrases=entity_phrases,
            )
            if semantic_answer:
                response = {
                    "triples_added": total_triples,
                    "query": semantic_cypher,
                    "results": semantic_results,
                    "answer": semantic_answer,
                    "steps_taken": [
                        {
                            "step": "semantic_lookup",
                            "terms": expanded_terms,
                            "matches": len(semantic_results),
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        path_cypher, path_results = graph.search_paths(expanded_terms, document_id, max_hops=3, limit=80)
        if should_use_multi_hop and path_results:
            path_answer = _format_path_results(path_results, expanded_terms, relation_hints)
            if path_answer:
                response = {
                    "triples_added": total_triples,
                    "query": path_cypher,
                    "results": path_results,
                    "answer": path_answer,
                    "steps_taken": [
                        {
                            "step": "multi_hop_lookup",
                            "terms": expanded_terms,
                            "relation_hints": sorted(relation_hints),
                            "matches": len(path_results),
                        }
                    ],
                }
                return _response_with_chunk_fallback(response, question, chunk_hits, graph_expanded_terms, relation_hints, entity_phrases)

        chunk_answer = _format_chunk_answer(
            question,
            chunk_hits,
            graph_expanded_terms,
            phrases=entity_phrases,
            relation_hints=relation_hints,
        )
        if chunk_answer:
            return {
                "triples_added": total_triples,
                "query": "SEMANTIC_CHUNK_LOOKUP",
                "results": [
                    {
                        "chunk": hit["chunk"],
                        "chunk_id": hit.get("chunk_id", ""),
                        "sentences": hit["sentences"],
                        "score": hit["score"],
                        "score_breakdown": {
                            "embedding": hit.get("embedding_score", 0),
                            "keyword": hit.get("keyword_score", 0),
                            "fuzzy": hit.get("fuzzy_score", 0),
                            "graph": hit.get("graph_score", 0),
                        },
                        "summary": hit.get("summary", ""),
                        "keywords": hit.get("keywords", []),
                        "section": hit.get("section", "Document"),
                        "page": hit.get("page"),
                    }
                    for hit in chunk_hits[:3]
                ],
                "answer": chunk_answer,
                "steps_taken": [
                    {
                        "step": "semantic_chunk_lookup",
                        "terms": graph_expanded_terms,
                        "matches": len(chunk_hits),
                    }
                ],
            }

        return {
            "triples_added": total_triples,
            "query": path_cypher if expanded_terms else direct_cypher,
            "results": [],
            "answer": "No data found in knowledge graph",
            "steps_taken": [
                {
                    "step": "no_match",
                    "terms": expanded_terms,
                    "matches": 0,
                }
            ],
        }
    finally:
        graph.close()
