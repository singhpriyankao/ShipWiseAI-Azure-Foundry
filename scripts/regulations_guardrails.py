"""
Deterministic guardrails for the Regulations agent. These run on the agent's output *after* it
responds, checking it against the same retrieved_chunks the agent was
given - never trusting the prompt alone to have gotten it right.

Highest-value one first: the citation validator. It defends the one thing
this agent exists to do - never citing a provision that wasn't actually
retrieved.
"""
import re


def _same(a, b):
    # The model sometimes writes a number as text ("1") where the index
    # stores an int (1). Compare as normalized strings so that isn't read
    # as a fabricated citation.
    return str(a).strip().lower() == str(b).strip().lower()


def validate_citations(agent_output, retrieved_chunks):
    """
    Checks every entry in agent_output["citations"] against retrieved_chunks.

    A citation field left as null means "any value is fine at this level",
    not "must be null" - e.g. citing Article 6 with paragraph_number=null is
    valid as long as *any* paragraph of Article 6 was retrieved, not only a
    chunk that itself has paragraph_number=null. This treats an imprecise
    but real citation differently from a fabricated one, which is the
    distinction this guardrail is actually meant to catch.

    Returns the list of citation entries that don't match anything in
    retrieved_chunks - empty list means every citation passed.
    """
    invalid = []
    for citation in agent_output.get("citations", []):
        if not _citation_matches_any_chunk(citation, retrieved_chunks):
            invalid.append(citation)
    return invalid


def _citation_matches_any_chunk(citation, retrieved_chunks):
    ctype = citation.get("type")

    for chunk in retrieved_chunks:
        if chunk.get("regulation") != citation.get("regulation"):
            continue

        if ctype == "article":
            if chunk.get("is_annex") or chunk.get("is_recital"):
                continue
            if not _same(chunk.get("article_number"), citation.get("article_number")):
                continue
            cited_para = citation.get("paragraph_number")
            if cited_para is not None and not _same(chunk.get("paragraph_number"), cited_para):
                continue
            return True

        elif ctype == "recital":
            if not chunk.get("is_recital"):
                continue
            if not _same(chunk.get("recital_number"), citation.get("recital_number")):
                continue
            return True

        elif ctype == "annex":
            if not chunk.get("is_annex"):
                continue
            if not _same(chunk.get("annex_id"), citation.get("annex_id")):
                continue
            cited_item = citation.get("annex_item_number")
            if cited_item is not None and not _same(chunk.get("annex_item_number"), cited_item):
                continue
            cited_sub = citation.get("annex_sub_point")
            if cited_sub is not None and not _same(chunk.get("annex_sub_point"), cited_sub):
                continue
            return True

    return False


def check_grounding_coverage(agent_output, retrieved_chunks):
    """
    Checks that grounding_status is consistent with what retrieval actually
    returned, instead of trusting the model's own claim.

    Two cases checked, both narrow and specific - not a general "does this
    look grounded" judgment call, which is the model's job, not code's:
    1. retrieved_chunks was empty -> the agent had nothing to reason over
       at all, so grounding_status must be "ungrounded_fallback". Claiming
       "grounded" here would mean the model answered from memory, which
       the prompt explicitly forbids.
    2. grounding_status says "grounded" but citations is empty -> the agent
       is claiming it found something real without pointing at anything
       specific. Even a "no obligation applies" conclusion should still
       cite the provisions it considered and ruled out (see the citation
       rules in regulations.md) - a grounded answer with zero citations at
       all is a real inconsistency, not just an imprecise one.

    Returns None if grounding_status is consistent, or a short string
    describing the violation if not.
    """
    grounding_status = agent_output.get("grounding_status")
    citations = agent_output.get("citations", [])

    if not retrieved_chunks and grounding_status != "ungrounded_fallback":
        return (
            f'retrieved_chunks was empty but grounding_status was '
            f'"{grounding_status}" - nothing was actually retrieved to ground an answer in.'
        )

    if grounding_status == "grounded" and not citations:
        return 'grounding_status was "grounded" but citations was empty - a grounded answer should point at something.'

    return None


def check_recital_only(agent_output):
    """
    Rejects an answer whose only support is a recital - recitals are
    interpretive commentary, not binding law (see the recital handling
    rule in regulations.md). If at least one citation is an article or
    annex provision, any recitals alongside it are fine as extra context.

    Doesn't need retrieved_chunks - this only depends on what the agent
    actually cited, not what was available to cite. Applies the same way
    regardless of classification (including "no_specific_obligation_
    identified") - a negative conclusion still needs to trace to real
    binding text (what was ruled out and why), not rest on interpretive
    commentary alone either.

    Returns None if there's at least one non-recital citation (or no
    citations at all, which is check_grounding_coverage's job instead),
    or a short string describing the violation if every citation present
    is a recital.
    """
    citations = agent_output.get("citations", [])
    if not citations:
        return None

    if all(c.get("type") == "recital" for c in citations):
        return "every citation is a recital - no binding provision supports this answer, only interpretive commentary."

    return None


_DEFINITIVE_APPLICABILITY_PHRASES = [
    "immediately applicable",
    "immediately enforceable",
    "already applicable",
    "already in effect",
    "currently applicable",
    "currently enforceable",
    "in effect now",
]

_TIMING_KEYWORDS = [
    "entry into force",
    "months after",
    "shall apply from",
    "shall become applicable",
    "with effect from",
    "applicable from",
]


def check_applicability_grounding(agent_output, retrieved_chunks):
    """
    Checks that a confident applicability claim is actually backed by
    timing language somewhere in retrieved_chunks, instead of trusting the
    model's own wording. This is a heuristic (keyword matching on free
    text), not a full semantic check - it can't verify a claimed date is
    *correct*, only that *something* about timing was actually retrieved
    to base a confident claim on. Built specifically to catch the failure
    mode already seen by hand in testing: the model asserting "Requirements
    are immediately applicable" when nothing retrieved said anything about
    timing at all.

    Returns None if applicability_status doesn't make a confident claim
    (e.g. it's the honest "no date found" answer), or if it does and some
    retrieved chunk actually contains timing language to back it up.
    Returns a short string describing the violation if it makes a
    confident claim with zero timing-related support anywhere in
    retrieved_chunks.
    """
    status_text = (agent_output.get("applicability_status") or "").lower()

    makes_confident_claim = any(phrase in status_text for phrase in _DEFINITIVE_APPLICABILITY_PHRASES)
    if not makes_confident_claim:
        return None

    has_timing_support = any(
        keyword in (chunk.get("text") or "").lower() or keyword in (chunk.get("parent_article_text") or "").lower()
        for chunk in retrieved_chunks
        for keyword in _TIMING_KEYWORDS
    )
    if not has_timing_support:
        return (
            f'applicability_status makes a confident claim ("{agent_output.get("applicability_status")}") '
            f'but no retrieved chunk contains any timing/phase-in language to support it.'
        )

    return None


_DEROGATION_OUTCOMES = {"exception_applies", "exception_does_not_apply", "unclear", "not_relevant"}


def check_derogation_addressed(agent_output, retrieved_chunks):
    """
    Article 6(3) is the derogation that can take an Annex III system out of
    the high-risk category. If it was retrieved and the answer says
    high_risk, the required `derogation_assessment` field must actually
    assess it, and Article 6 paragraph 3 must be in `citations`. Everything
    checked is structured (enum value, non-empty reason, citation fields),
    so it's plain field checks like the citation validator, not reading prose.

    Why a field and not a prose sentence: the first fix was a prompt rule
    asking for a sentence beginning "Article 6(3) derogation:". It was
    ignored in 3 of 3 completed test runs and the exception was skipped
    outright in 4 of 4 runs before that. A required schema field is much
    harder for the model to leave out.

    Limitation: this proves the assessment exists and is self-consistent,
    not that the reasoning in `reason` is sound. "unclear" is accepted on
    purpose - it's an honest answer, and forcing yes/no would just push the
    model to guess.

    Returns None if there's nothing to check or it was addressed, or a short
    string describing the violation.
    """
    if agent_output.get("classification") != "high_risk":
        return None

    retrieved_6_3 = any(
        not c.get("is_annex") and not c.get("is_recital")
        and _same(c.get("article_number"), 6) and _same(c.get("paragraph_number"), 3)
        for c in retrieved_chunks
    )
    if not retrieved_6_3:
        return None

    assessment = agent_output.get("derogation_assessment")
    if not isinstance(assessment, dict):
        return "Article 6(3) was retrieved and the answer says high_risk, but derogation_assessment is missing."

    outcome = assessment.get("outcome")
    if outcome not in _DEROGATION_OUTCOMES:
        return f'derogation_assessment.outcome "{outcome}" is not one of {sorted(_DEROGATION_OUTCOMES)}.'
    if outcome == "not_relevant":
        return 'Article 6(3) was retrieved and the answer says high_risk, but derogation_assessment.outcome is "not_relevant".'
    if outcome == "exception_applies":
        return 'derogation_assessment says "exception_applies" but classification is "high_risk" - these contradict each other.'
    if not (assessment.get("reason") or "").strip():
        return "derogation_assessment.reason is empty."

    cited_6_3 = any(
        c.get("type") == "article" and _same(c.get("article_number"), 6) and _same(c.get("paragraph_number"), 3)
        for c in agent_output.get("citations", [])
    )
    if not cited_6_3:
        return "derogation_assessment is filled in but Article 6(3) is not in citations."

    return None


def add_missing_derogation_citation(agent_output, retrieved_chunks):
    """
    If the model filled in a usable derogation_assessment but left Article
    6(3) out of `citations`, add the citation in code. The correct value is
    already known (the assessment is the model saying it used 6(3), and we
    know exactly which retrieved chunk that is), so a retry would only spend
    another agent call. In v7 testing this omission caused every guardrail
    failure (3 failed attempts, one run ended analysis_failed) on answers
    whose reasoning was otherwise fine.

    Only adds when the assessment would otherwise pass check_derogation_
    addressed - never papers over a missing/empty/contradictory assessment.
    The citation's fields are copied from the retrieved chunk, not written
    from assumption. Returns True if a citation was added.
    """
    if agent_output.get("classification") != "high_risk":
        return False

    chunk_6_3 = next(
        (c for c in retrieved_chunks
         if not c.get("is_annex") and not c.get("is_recital")
         and _same(c.get("article_number"), 6) and _same(c.get("paragraph_number"), 3)),
        None,
    )
    if chunk_6_3 is None:
        return False

    assessment = agent_output.get("derogation_assessment")
    if not isinstance(assessment, dict):
        return False
    if assessment.get("outcome") not in ("exception_does_not_apply", "unclear"):
        return False
    if not (assessment.get("reason") or "").strip():
        return False

    citations = agent_output.setdefault("citations", [])
    already_cited = any(
        c.get("type") == "article" and _same(c.get("article_number"), 6) and _same(c.get("paragraph_number"), 3)
        for c in citations
    )
    if already_cited:
        return False

    citations.append({
        "type": "article",
        "article_number": chunk_6_3.get("article_number"),
        "paragraph_number": chunk_6_3.get("paragraph_number"),
        "recital_number": None,
        "annex_id": None,
        "annex_item_number": None,
        "annex_sub_point": None,
        "regulation": chunk_6_3.get("regulation"),
    })
    return True


_CHAPTER_II_CLAUSE = re.compile(
    r"Chapters I and II shall apply from (\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
    re.I,
)


def prohibition_applicability_status(retrieved_chunks):
    """
    A `prohibited` finding always rests on Article 5, which sits in Chapter
    II, and Article 113 gives Chapters I and II an earlier application date
    than the Act's general one. In testing the model said a prohibition
    applied "from 2 August 2026" - a real date from Article 113, attached to
    the wrong provision, so check_applicability_dates passed it. For the
    highest-severity classification that's too risky to leave to the model.

    Reads the Chapter II date out of the retrieved Article 113 text instead
    of hard-coding it, so it stays tied to the ingested Act. Returns the
    status string, or None if the clause isn't found (caller then falls back
    to a pointer at Article 113).
    """
    for c in retrieved_chunks:
        text = (c.get("text") or "") + " " + (c.get("parent_article_text") or "")
        if _same(c.get("article_number"), 113):
            m = _CHAPTER_II_CLAUSE.search(text)
            if m:
                return (
                    f"The prohibitions in Article 5 (Chapter II) apply from {m.group(1)}, "
                    f"per Article 113(a), earlier than the Act's general application date."
                )
    return None


_DATE = r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})"
_ART_113_CLAUSES = {
    "general": re.compile(r"It shall apply from " + _DATE, re.I),
    "a": re.compile(r"Chapters I and II shall apply from " + _DATE, re.I),
    "b": re.compile(r"Chapter III Section 4, Chapter V, Chapter VII and Chapter XII and Article 78 shall apply from " + _DATE, re.I),
    "c": re.compile(r"Article 6\(1\) and the corresponding obligations[^.;]*? shall apply from " + _DATE, re.I),
}
_CLAUSE_LABEL = {
    "general": "the general application date",
    "a": "Article 113(a)",
    "b": "Article 113(b)",
    "c": "Article 113(c)",
}


def _article_bucket(article_number, paragraph_number):
    """Which Article 113 date applies to an article, or None if not mapped.
    Chapter layout is the Act's own: I = Arts 1-4, II = 5, III = 6-49 (Section 4
    = 28-39), IV = 50, V = 51-56, VI = 57-63, VII = 64-70, VIII = 71,
    IX = 72-94, X = 95-96, XI = 97-98, XII = 99-101, XIII = 102-113. Article
    78 and Article 101 are the named exceptions in 113(b). Anything past 101
    (transitional and amending articles) is left unmapped on purpose."""
    try:
        n = int(str(article_number).strip())
    except ValueError:
        return None
    if n == 6 and _same(paragraph_number, 1):
        return "c"
    if 1 <= n <= 5:
        return "a"
    if 28 <= n <= 39 or 51 <= n <= 56 or 64 <= n <= 70 or n == 78 or 99 <= n <= 100:
        return "b"
    if 6 <= n <= 101:
        return "general"
    return None


def derive_applicability_status(agent_output, retrieved_chunks):
    """
    Builds the applicability_status in code from the provisions the answer
    cites and the Article 113 text that was retrieved, instead of trusting
    the model to match dates to provisions. Found in a live run: the model
    said Article 50(3) applies from 2 August 2025 - a real date in Article
    113, but it belongs to Chapters III Section 4, V, VII and XII; Article 50
    is Chapter IV, so the general date applies. check_applicability_dates
    passed it because the date does appear in the retrieved text.

    Dates are read out of the retrieved Article 113, not hard-coded. The
    wording says what Article 113 gives each cited provision, not that the
    provision applies to the feature, because citations also include
    provisions the analysis ruled out.

    Returns: a status string; "" if the answer cites no EU AI Act article or
    annex (nothing to date); or None if a cited provision can't be mapped or
    the Article 113 clauses can't be read (caller falls back to a pointer).
    Recitals and other regulations (GDPR) are ignored. Only Annex III is
    mapped (it takes effect through Article 6(2), so the general date).
    """
    art_113 = ""
    for c in retrieved_chunks:
        if _same(c.get("article_number"), 113):
            art_113 += " " + (c.get("text") or "") + " " + (c.get("parent_article_text") or "")

    by_bucket = {}
    for cit in agent_output.get("citations", []):
        if cit.get("regulation") not in (None, "EU_AI_Act"):
            continue
        if cit.get("type") == "recital":
            continue
        if cit.get("type") == "annex":
            if not _same(cit.get("annex_id"), "III"):
                return None
            label, bucket = "Annex III", "general"
        elif cit.get("article_number") is not None:
            if _same(cit.get("article_number"), 113):
                continue
            bucket = _article_bucket(cit.get("article_number"), cit.get("paragraph_number"))
            if bucket is None:
                return None
            label = f"Article {str(cit['article_number']).strip()}"
            if bucket == "c":
                label += "(1)"
        else:
            continue
        by_bucket.setdefault(bucket, [])
        if label not in by_bucket[bucket]:
            by_bucket[bucket].append(label)

    if not by_bucket:
        return ""

    parts = []
    for bucket in ("a", "b", "general", "c"):
        if bucket not in by_bucket:
            continue
        m = _ART_113_CLAUSES[bucket].search(art_113)
        if not m:
            return None
        parts.append(f"{m.group(1)} ({', '.join(by_bucket[bucket])}; {_CLAUSE_LABEL[bucket]})")
    return "Under Article 113, the provisions cited here take effect on these dates: " + "; ".join(parts) + "."


def mark_insufficient_information(agent_output):
    """
    Turns a non-answer into an explicit non-answer. In testing, a description
    too vague to assess ("add some AI to our product") and an out-of-scope
    question both came back with classification
    "no_specific_obligation_identified" - which reads as "you're in the
    clear" for a feature nobody described.

    Triggers when grounding_status is "ungrounded_fallback" (the prompt says
    not to force a classification then), or when missing_inputs is non-empty
    AND the classification is "no_specific_obligation_identified". It
    deliberately does NOT override an adverse classification (high_risk,
    prohibited, transparency_obligated) just because missing_inputs is
    filled in: wrongly telling a PM "you're in the clear" is the dangerous
    error; wrongly keeping a caution is the safe one.

    Mutates agent_output in place (classification becomes
    "insufficient_information", the conclusion-bearing fields become
    "not_assessed", citations are cleared since they supported no
    conclusion). Returns True if it changed anything.
    """
    ungrounded = agent_output.get("grounding_status") == "ungrounded_fallback"
    vague_all_clear = (
        bool(agent_output.get("missing_inputs"))
        and agent_output.get("classification") == "no_specific_obligation_identified"
    )
    if not (ungrounded or vague_all_clear):
        return False

    agent_output["classification"] = "insufficient_information"
    agent_output["derogation_assessment"] = {"outcome": "not_assessed", "reason": ""}
    agent_output["applicability_status"] = "not_assessed"
    agent_output["citations"] = []
    return True


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_NUM = {m.lower(): i for i, m in enumerate(_MONTHS.split("|"), start=1)}
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_MY = re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(20\d\d)\b")

# Phrases that mean "I couldn't determine a date" - an honest answer, allowed
# to pass without any date in it.
_NO_DATE_PHRASES = ["could not be determined", "could not be verified", "no applicability date found"]


def _extract_dates(text, include_bare_years):
    """Returns a set of (year, month, day) tuples; month/day are None when
    the text only gave month+year or a bare year."""
    found = set()
    for m in _DMY.finditer(text):
        found.add((int(m.group(3)), _MONTH_NUM[m.group(2).lower()], int(m.group(1))))
    text = _DMY.sub(" ", text)
    for m in _MDY.finditer(text):
        found.add((int(m.group(3)), _MONTH_NUM[m.group(1).lower()], int(m.group(2))))
    text = _MDY.sub(" ", text)
    for m in _MY.finditer(text):
        found.add((int(m.group(2)), _MONTH_NUM[m.group(1).lower()], None))
    text = _MY.sub(" ", text)
    if include_bare_years:
        for m in _YEAR.finditer(text):
            found.add((int(m.group(1)), None, None))
    return found


def check_applicability_dates(agent_output, retrieved_chunks):
    """
    Every date in applicability_status must appear in the retrieved text,
    and the status must contain at least one date unless it says plainly
    that no date could be determined. Built for the real failures seen in
    repeat testing: a date the model supplied from memory ("July 12, 2024"
    appears nowhere in the ingested Act) and a generic "obligations are
    applicable" with no date at all.

    Bare years in the retrieved text are not counted as dates ("2024" shows
    up in "Regulation (EU) 2024/1689"); only full dates and month+year are.

    Limitations: it catches invented dates, not a real date attached to the
    wrong provision (e.g. a true date from one article described as coming
    from another). It only recognizes written-out dates ("2 August 2026"),
    so an ISO or numeric date is treated as "no date" and fails safe.

    Returns None if acceptable, or a short string describing the problem.
    """
    status = agent_output.get("applicability_status") or ""
    lowered = status.lower()
    if any(p in lowered for p in _NO_DATE_PHRASES):
        return None

    claimed = _extract_dates(status, include_bare_years=True)
    if not claimed:
        return f'applicability_status contains no date and does not say a date could not be determined: "{status}"'

    source_text = " ".join((c.get("text") or "") + " " + (c.get("parent_article_text") or "") for c in retrieved_chunks)
    available = _extract_dates(source_text, include_bare_years=False)

    unverified = []
    for y, mo, d in claimed:
        if d is not None:
            ok = (y, mo, d) in available
        elif mo is not None:
            ok = any(ay == y and am == mo for ay, am, ad in available)
        else:
            ok = any(ay == y for ay, am, ad in available)
        if not ok:
            unverified.append((y, mo, d))

    if unverified:
        return f"applicability_status states date(s) not found in any retrieved text: {sorted(unverified, key=str)}"

    return None
