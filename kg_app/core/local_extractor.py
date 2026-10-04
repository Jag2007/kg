"""
Local knowledge extraction with spaCy.

No API calls: entities come from NER + noun chunks, relations come from
dependency patterns (subject-verb-object, passive voice, copula, verb +
preposition, appositives). Every chunk also reports which entities it
mentions so the graph can link chunks <-> entities for multi-hop search.
"""

import os
import re
import threading

from kg_app.core.extractor import clean_and_validate_triples

_NLP = None
_NLP_LOCK = threading.Lock()

SKIPPED_ENTITY_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS = {"dobj", "attr", "oprd", "dative"}
PRONOUNS = {"it", "they", "he", "she", "this", "these", "those", "its", "their", "his", "her", "which", "who", "that"}
LEADING_NOISE = re.compile(r"^(the|a|an|this|that|these|those|its|their|his|her|our|your|my|some|any|each|every)\s+", re.IGNORECASE)
MAX_ENTITY_WORDS = 6
MAX_SENTENCE_ENTITIES = 10
MAX_CHARS = int(os.environ.get("KG_SPACY_MAX_CHARS", "200000"))

# Verbs that read badly as relations on their own get folded into a clearer name.
VERB_RELATION_OVERRIDES = {
    "be": "IS",
    "have": "HAS",
    "include": "INCLUDES",
    "contain": "CONTAINS",
    "consist": "CONSISTS_OF",
    "locate": "LOCATED_IN",
    "base": "BASED_IN",
    "headquarter": "HEADQUARTERED_IN",
    "found": "FOUNDED",
    "establish": "ESTABLISHED",
    "create": "CREATED",
    "develop": "DEVELOPED",
    "invent": "INVENTED",
    "write": "WROTE",
    "lead": "LEADS",
    "own": "OWNS",
    "use": "USES",
    "run": "RUNS",
    "require": "REQUIRES",
    "support": "SUPPORTS",
    "produce": "PRODUCES",
    "cause": "CAUSES",
    "acquire": "ACQUIRED",
    "join": "JOINED",
    "manage": "MANAGES",
    "serve": "SERVES",
    "connect": "CONNECTS_TO",
    "depend": "DEPENDS_ON",
    "listen": "LISTENS_ON",
}

ROLE_WORDS = {
    "ceo", "cto", "cfo", "founder", "cofounder", "co-founder", "president", "director", "head",
    "leader", "chair", "chairman", "manager", "member", "author", "owner", "capital", "headquarters",
    "part", "professor", "student", "employee", "engineer", "partner", "mayor", "minister",
}


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    with _NLP_LOCK:
        if _NLP is None:
            import spacy

            model_name = os.environ.get("KG_SPACY_MODEL", "en_core_web_sm")
            try:
                _NLP = spacy.load(model_name)
            except OSError as error:
                raise RuntimeError(
                    f"spaCy model '{model_name}' is not installed. "
                    f"Run: python -m spacy download {model_name}"
                ) from error
            _NLP.max_length = max(_NLP.max_length, MAX_CHARS + 1000)
    return _NLP


def is_available() -> bool:
    try:
        _get_nlp()
        return True
    except Exception:
        return False


def _clean_name(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\n,;:.()[]{}\"'")
    value = re.sub(r"'s$|’s$", "", value).strip()
    previous = None
    while previous != value:
        previous = value
        value = LEADING_NOISE.sub("", value).strip()
    if not value or len(value) < 2:
        return ""
    if not re.search(r"[A-Za-z]", value) and not re.fullmatch(r"(1[5-9]|20)\d\d", value):
        return ""
    if len(value.split()) > MAX_ENTITY_WORDS:
        return ""
    if value.lower() in PRONOUNS:
        return ""
    return value


class _SentenceEntities:
    """Maps token indexes to the entity span that covers them."""

    def __init__(self, sent):
        self.by_token: dict[int, str] = {}
        self.names: list[str] = []
        spans = []
        root = sent.root.i
        for ent in sent.ents:
            if ent.label_ in SKIPPED_ENTITY_LABELS:
                continue
            start, end = ent.start, ent.end
            if start < root < end or (end - 1 == root and end - start > 1):
                # NER swallowed the predicate ("Backend A runs") - cut it off.
                end = root
            if end > start:
                spans.append((start, end, sent.doc[start:end].text))
        for chunk in sent.noun_chunks:
            if chunk.root.i == root:
                continue
            if chunk.root.pos_ in {"PRON", "DET", "NUM"}:
                continue
            if any(token.pos_ in {"VERB", "CCONJ"} for token in chunk) and not chunk.ents:
                continue
            spans.append((chunk.start, chunk.end, chunk.text))
        # Longer spans first so "Helix Labs Berlin office" beats "Berlin".
        for start, end, text in sorted(spans, key=lambda item: item[0] - item[1]):
            name = _clean_name(text)
            if not name:
                continue
            free = [i for i in range(start, end) if i not in self.by_token]
            if not free:
                continue
            for i in free:
                self.by_token[i] = name
            if name.lower() not in {n.lower() for n in self.names}:
                self.names.append(name)

    def of(self, token) -> str:
        name = ""
        if token.i in self.by_token:
            name = self.by_token[token.i]
        elif token.pos_ in {"NOUN", "PROPN"}:
            parts = [child.text for child in token.lefts if child.dep_ in {"compound", "amod"}]
            name = _clean_name(" ".join(parts + [token.text]))
        # Keep identifiers attached: "port 443", "Backend 2", "version 3.1".
        numbers = [child.text for child in token.rights if child.dep_ in {"nummod", "appos"} and child.like_num]
        if name and numbers and numbers[0] not in name:
            name = f"{name} {numbers[0]}"
        return name


def _expand_conjuncts(token) -> list:
    items = [token]
    for child in token.children:
        if child.dep_ == "conj":
            items.extend(_expand_conjuncts(child))
    return items


def _is_negated(verb) -> bool:
    return any(child.dep_ == "neg" for child in verb.children)


def _relation_name(verb, prep=None, passive: bool = False) -> str:
    lemma = verb.lemma_.lower()
    inflected = verb.text.upper() if verb.tag_ in {"VBZ", "VBD"} else lemma.upper()
    base = VERB_RELATION_OVERRIDES.get(lemma, inflected)
    if passive:
        # "X was founded by Y" -> X FOUNDED_BY Y
        if base.endswith("_IN") or base.endswith("_ON"):
            return base
        participle = verb.text.upper() if verb.tag_ == "VBN" else base
        return f"{participle}_BY" if prep is None or prep.lower_ == "by" else f"{participle}_{prep.text.upper()}"
    if prep is not None and "_" not in base:
        return f"{base}_{prep.text.upper()}"
    return base


def _subjects_of(verb, sentence_entities, last_subject: str) -> tuple[list[str], bool]:
    subjects: list[str] = []
    passive = False
    head = verb
    # Conjoined verbs ("X founded A and runs B") share the first verb's subject.
    while not any(child.dep_ in SUBJECT_DEPS for child in head.children) and head.dep_ == "conj":
        head = head.head
    for child in head.children:
        if child.dep_ not in SUBJECT_DEPS:
            continue
        passive = passive or child.dep_ == "nsubjpass"
        for item in _expand_conjuncts(child):
            if item.lower_ in {"which", "who", "that"} and verb.dep_ == "relcl":
                name = sentence_entities.of(verb.head)
                if name:
                    subjects.append(name)
                continue
            if item.lower_ in PRONOUNS and last_subject:
                subjects.append(last_subject)
                continue
            name = sentence_entities.of(item)
            if name:
                subjects.append(name)
    # Mis-parsed "nginx listens on ..." (verb tagged as a noun with the subject as its compound).
    if not subjects and verb.dep_ == "ROOT" and verb.pos_ != "VERB":
        lefts = [child for child in verb.lefts if child.dep_ in {"compound", "amod", "nmod"}]
        if lefts:
            span = verb.doc[lefts[0].left_edge.i : verb.i]
            name = _clean_name(span.text)
            if name:
                subjects.append(name)
    # Relative clause: "Helix Labs, which is based in Berlin" -> subject is the head noun.
    if not subjects and verb.dep_ == "relcl":
        name = sentence_entities.of(verb.head)
        if name:
            subjects.append(name)
    return subjects, passive


def _triples_for_verb(verb, sentence_entities, last_subject: str) -> list[dict]:
    if _is_negated(verb):
        return []
    subjects, passive = _subjects_of(verb, sentence_entities, last_subject)
    if not subjects:
        return []

    triples: list[dict] = []

    def add(subject: str, relation: str, obj: str, confidence: float) -> None:
        if subject and obj and subject.lower() != obj.lower():
            triples.append({"subject": subject, "relation": relation, "object": obj, "confidence": confidence})

    for child in verb.children:
        if child.dep_ in OBJECT_DEPS or (child.dep_ == "acomp" and verb.lemma_ == "be"):
            for item in _expand_conjuncts(child):
                obj = sentence_entities.of(item)
                if verb.lemma_ == "be":
                    relation = "IS_A" if child.dep_ == "attr" else ""
                    # "X is responsible for Y" / "X is located in Y"
                    for prep in item.children:
                        if prep.dep_ == "prep":
                            for pobj in prep.children:
                                if pobj.dep_ == "pobj":
                                    for target in _expand_conjuncts(pobj):
                                        rel = f"{item.lemma_.upper()}_{prep.text.upper()}"
                                        if item.lower_ in ROLE_WORDS:
                                            rel = f"{item.text.upper()}_{prep.text.upper()}"
                                        for subject in subjects:
                                            add(subject, rel, sentence_entities.of(target), 0.75)
                    if relation and obj:
                        for subject in subjects:
                            add(subject, relation, obj, 0.75)
                    continue
                for subject in subjects:
                    add(subject, _relation_name(verb), obj, 0.8)
        elif child.dep_ in {"prep", "agent"}:
            for pobj in child.children:
                if pobj.dep_ != "pobj":
                    continue
                for item in _expand_conjuncts(pobj):
                    obj = sentence_entities.of(item)
                    if passive:
                        relation = _relation_name(verb, child, passive=True)
                        confidence = 0.82
                    else:
                        relation = _relation_name(verb, child)
                        confidence = 0.72
                    for subject in subjects:
                        add(subject, relation, obj, confidence)
    return triples


def _appositive_triples(sent, sentence_entities) -> list[dict]:
    """'Maya Rao, the founder of Helix Labs, ...' -> Maya Rao FOUNDER_OF Helix Labs."""
    triples = []
    for token in sent:
        if token.dep_ != "appos":
            continue
        head_name = sentence_entities.of(token.head)
        if not head_name:
            continue
        role = token.lemma_.lower()
        linked = False
        for prep in token.children:
            if prep.dep_ != "prep":
                continue
            for pobj in prep.children:
                if pobj.dep_ == "pobj":
                    target = sentence_entities.of(pobj)
                    if target:
                        triples.append({
                            "subject": head_name,
                            "relation": f"{role.upper()}_{prep.text.upper()}",
                            "object": target,
                            "confidence": 0.78,
                        })
                        linked = True
        if not linked:
            target = sentence_entities.of(token)
            if target and target.lower() != head_name.lower():
                triples.append({"subject": head_name, "relation": "IS_A", "object": target, "confidence": 0.7})
    return triples


def _noun_prep_triples(sent, sentence_entities) -> list[dict]:
    """'the capital of France' / 'the CEO of Helix' -> role relations between named things."""
    triples = []
    for token in sent:
        if token.dep_ != "prep" or token.head.pos_ not in {"NOUN", "PROPN"}:
            continue
        head = token.head
        if head.lower_ not in ROLE_WORDS:
            continue
        owner = head.head if head.dep_ in {"attr", "appos"} else None
        for pobj in token.children:
            if pobj.dep_ != "pobj" or owner is None:
                continue
            subject = sentence_entities.of(owner) if owner.pos_ in {"NOUN", "PROPN"} else ""
            if not subject:
                continue
            triples.append({
                "subject": subject,
                "relation": f"{head.text.upper()}_{token.text.upper()}",
                "object": sentence_entities.of(pobj),
                "confidence": 0.7,
            })
    return [t for t in triples if t["object"]]


def _relation_is_a_name(triple: dict, source: str) -> bool:
    """Drop 'DNS server DHANVIN_ON Mac 1' - the 'verb' is someone's name from '(Dhanvin on Mac 1)'."""
    word = str(triple.get("relation", "")).split("_")[0]
    if len(word) < 3:
        return False
    context = triple.get("sentence") or source
    if re.search(rf"\b{re.escape(word.lower())}\b", context):
        return False
    # A capitalised word mid-sentence is a name; at a sentence start it may just be an imperative verb.
    for match in re.finditer(rf"\b{re.escape(word.capitalize())}\b", context):
        before = context[: match.start()].rstrip()
        if before and before[-1] not in ".!?:\n":
            return True
    return False


def extract_chunk_knowledge(text: str) -> dict:
    """Return {'triples': [...], 'entities': [...]} for one chunk of text."""
    source = str(text or "").strip()
    if len(source) < 10:
        return {"triples": [], "entities": []}

    nlp = _get_nlp()
    doc = nlp(source[:MAX_CHARS])
    raw_triples: list[dict] = []
    entities: list[str] = []
    seen_entities: set[str] = set()
    last_subject = ""

    for sent in doc.sents:
        sentence_entities = _SentenceEntities(sent)
        for name in sentence_entities.names[:MAX_SENTENCE_ENTITIES]:
            if name.lower() not in seen_entities:
                seen_entities.add(name.lower())
                entities.append(name)

        sentence_triples: list[dict] = []
        for token in sent:
            has_subject = any(child.dep_ in SUBJECT_DEPS for child in token.children)
            looks_like_verb = token.tag_ in {"VBZ", "VBD", "VBP", "VB", "NNS"} and token.pos_ != "PROPN"
            is_predicate = token.pos_ == "VERB" or token.lemma_ == "be" or (
                token.dep_ == "ROOT"
                and looks_like_verb
                and (has_subject or any(c.dep_ in {"prep", "dobj"} for c in token.children))
            )
            if is_predicate:
                sentence_triples.extend(_triples_for_verb(token, sentence_entities, last_subject))
        sentence_triples.extend(_appositive_triples(sent, sentence_entities))
        sentence_triples.extend(_noun_prep_triples(sent, sentence_entities))

        for triple in sentence_triples:
            triple["sentence"] = sent.text.strip()[:400]
        raw_triples.extend(sentence_triples)

        # Remember the main subject for pronoun carry-over into the next sentence.
        for token in sent:
            if token.dep_ in {"nsubj", "nsubjpass"} and token.lower_ not in PRONOUNS:
                name = sentence_entities.of(token)
                if name:
                    last_subject = name
                    break

    # Regex heuristics catch short technical statements the small parser misreads.
    from kg_app.core.extractor import extract_triples_fallback

    known = {(t["subject"].lower(), t["object"].lower()) for t in raw_triples}
    for triple in extract_triples_fallback(source):
        if (triple["subject"].lower(), triple["object"].lower()) not in known:
            triple.setdefault("confidence", 0.55)
            raw_triples.append(triple)

    raw_triples = [triple for triple in raw_triples if not _relation_is_a_name(triple, source)]
    cleaned = clean_and_validate_triples(raw_triples)
    # clean_and_validate_triples drops extra keys; re-attach confidence + sentence evidence.
    evidence = {}
    for triple in raw_triples:
        key = (triple["subject"].lower(), triple["relation"].upper(), triple["object"].lower())
        evidence.setdefault(key, triple)
    for triple in cleaned:
        key = (triple["subject"].lower(), triple["relation"].upper(), triple["object"].lower())
        source_triple = evidence.get(key)
        if source_triple:
            triple["confidence"] = source_triple.get("confidence", 0.6)
            triple["sentence"] = source_triple.get("sentence", "")
    return {"triples": cleaned, "entities": entities}


def extract_triples_local(text: str) -> list[dict]:
    return extract_chunk_knowledge(text)["triples"]
