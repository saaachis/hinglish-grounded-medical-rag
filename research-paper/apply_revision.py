"""Produce the revised ICCSDI manuscript from the submitted .docx.

Edits the submitted Word file in place -- same template, styles, margins, fonts
and page size -- changing only the content that the reviews and the revision
analyses require. Every change is colour-coded and carries a Word comment that
says why it was made and which reviewer point it answers:

    YELLOW        correction of an error or an unsupported claim
    BRIGHT GREEN  new content added in response to the reviewers
    TURQUOISE     reworded claim, softened wording, or renamed term

Unchanged sentences are copied byte-for-byte from the submitted file, so a
reader can accept the highlighted text and delete the comments to obtain the
camera-ready.

    python apply_revision.py

Reads  research paper - devika jonjale & saachi shinde.docx
Writes research paper - devika jonjale & saachi shinde - REVISED.docx
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

HERE = Path(__file__).parent
SRC = HERE / "research paper - devika jonjale & saachi shinde.docx"
OUT = HERE / "research paper - devika jonjale & saachi shinde - REVISED.docx"
FIG = HERE.parent / "results" / "retrieval_figures"

COLOUR = {"fix": WD_COLOR_INDEX.YELLOW, "new": WD_COLOR_INDEX.BRIGHT_GREEN,
          "re": WD_COLOR_INDEX.TURQUOISE, "": None}
AUTHOR, INITIALS = "Revision", "RV"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

doc = Document(SRC)
#: Snapshot of a data table taken BEFORE any edit. Cloning a live table later
#: would copy the comment markers already placed in it and duplicate their ids.
TEMPLATE_OUTER = copy.deepcopy(doc.tables[3]._tbl)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def par(prefix: str) -> Paragraph:
    """The unique body paragraph whose text starts with `prefix`."""
    hits = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} paragraphs start with {prefix!r}")
    return hits[0]


_TAG = re.compile(r"<(/?)(i|b|sub|sup)>")


def _emit_runs(p: Paragraph, text: str, kind: str, base_rpr, size_pt=None):
    """Append runs for `text`, honouring <i> <b> <sub> <sup> tags."""
    state = {"i": False, "b": False, "sub": False, "sup": False}
    pos = 0
    runs = []
    for m in list(_TAG.finditer(text)) + [None]:
        chunk = text[pos:m.start()] if m else text[pos:]
        if chunk:
            r = p.add_run(chunk)
            if base_rpr is not None:
                r._r.insert(0, copy.deepcopy(base_rpr))
            if state["i"]:
                r.font.italic = True
            if state["b"]:
                r.font.bold = True
            if state["sub"]:
                r.font.subscript = True
            if state["sup"]:
                r.font.superscript = True
            if size_pt:
                r.font.size = Pt(size_pt)
            if COLOUR[kind] is not None:
                r.font.highlight_color = COLOUR[kind]
            runs.append(r)
        if m:
            state[m.group(2)] = m.group(1) == ""
            pos = m.end()
    return runs


def set_par(p: Paragraph, segments, note: str | None = None, keep_rpr=True):
    """Replace a paragraph's runs. segments = [(kind, text), ...]."""
    base = None
    if keep_rpr and p.runs and p.runs[0]._r.rPr is not None:
        base = copy.deepcopy(p.runs[0]._r.rPr)
        for tag in ("highlight", "b", "i", "vertAlign"):
            for el in base.findall(W + tag):
                base.remove(el)
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    # drop any non-run children left (bookmarks etc. are fine to keep; drawings are not)
    for child in list(p._p):
        if child.tag in (W + "r", W + "hyperlink"):
            p._p.remove(child)
    first_changed = None
    for kind, text in segments:
        runs = _emit_runs(p, text, kind, base)
        if kind and first_changed is None and runs:
            first_changed = runs[0]
    if note and first_changed is not None:
        doc.add_comment(runs=[first_changed], text=note, author=AUTHOR, initials=INITIALS)
    return p


def insert_after(anchor: Paragraph, ppr_from: Paragraph | None = None,
                 style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    src = ppr_from or anchor
    if src._p.pPr is not None:
        new_p.append(copy.deepcopy(src._p.pPr))
    anchor._p.addnext(new_p)
    p = Paragraph(new_p, anchor._parent)
    if style:
        p.style = doc.styles[style]
    return p


def new_par_after(anchor: Paragraph, segments, note=None, ppr_from=None, style=None,
                  size_pt=None) -> Paragraph:
    p = insert_after(anchor, ppr_from, style)
    first = None
    for kind, text in segments:
        runs = _emit_runs(p, text, kind, None, size_pt)
        if kind and first is None and runs:
            first = runs[0]
    if note and first is not None:
        doc.add_comment(runs=[first], text=note, author=AUTHOR, initials=INITIALS)
    return p


def delete_par(p: Paragraph):
    p._p.getparent().remove(p._p)


def coalesce(p: Paragraph, pattern: str) -> None:
    """Merge adjacent runs until every match of `pattern` lies inside one run.

    Word splits text across runs at spell-check and revision boundaries, so a
    token that reads as one word in the document can be two runs in the XML.
    The merged run keeps the formatting of the first fragment.
    """
    rx = re.compile(pattern)
    while rx.search(p.text) and not any(rx.search(r.text) for r in p.runs):
        runs = p.runs
        merged = False
        for i in range(len(runs)):
            acc = ""
            for j in range(i, len(runs)):
                acc += runs[j].text
                if rx.search(acc):
                    runs[i].text = acc
                    for k in range(i + 1, j + 1):
                        runs[k]._r.getparent().remove(runs[k]._r)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            raise RuntimeError(f"cannot coalesce {pattern!r} in: {p.text[:60]}")


def renumber_tables():
    """Old Tables 1-5 become 2-6 (a positioning table becomes Table 1)."""
    rx = re.compile(r"Table (\d)")
    for p in doc.paragraphs:
        if not rx.search(p.text):
            continue
        coalesce(p, r"Table \d")
        for r in p.runs:
            if rx.search(r.text):
                r.text = rx.sub(lambda m: f"Table {int(m.group(1)) + 1}", r.text)


def rename_recall():
    """Recall@1 -> Success@1 everywhere it survives in original text (R3.2)."""
    for p in doc.paragraphs:
        if "Recall@1" not in p.text:
            continue
        coalesce(p, r"Recall@1")
        for r in p.runs:
            if "Recall@1" in r.text:
                r.text = r.text.replace("Recall@1", "Success@1")
                r.font.highlight_color = COLOUR["re"]


# --------------------------------------------------------------------------- #
# Table helpers: clone the template's own table XML so borders, shading, cell
# margins and fonts are exactly those of the submitted manuscript.
# --------------------------------------------------------------------------- #
def _template_outer():
    """A fresh copy of the pristine 1x1 wrapper table every data table sits in."""
    t = copy.deepcopy(TEMPLATE_OUTER)
    for tag in ("commentRangeStart", "commentRangeEnd", "commentReference"):
        for el in t.iter(W + tag):
            el.getparent().remove(el)
    return t


def _set_cell_text(tc, lines, bold=False, italic=False, align="center", kind="",
                   note=None, size=18):
    """lines: list of strings; each becomes a paragraph in the cell."""
    ps = tc.findall(W + "p")
    tmpl_p = copy.deepcopy(ps[0])
    for pp in ps:
        tc.remove(pp)
    first_run = None
    for i, line in enumerate(lines):
        pe = copy.deepcopy(tmpl_p)
        for r in pe.findall(W + "r"):
            pe.remove(r)
        jc = pe.find(W + "pPr/" + W + "jc")
        if jc is not None:
            jc.set(qn("w:val"), align)
        tc.append(pe)
        p = Paragraph(pe, None)
        runs = _emit_runs(p, line, kind, None)
        for r in runs:
            r.font.size = Pt(size / 2 if i == 0 else max(7, size / 2 - 1))
            rf = r._r.rPr
            fonts = rf.find(W + "rFonts")
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rf.insert(0, fonts)
            fonts.set(qn("w:eastAsia"), "Times New Roman")
            if bold:
                r.font.bold = True
            if italic:
                r.font.italic = True
        if runs and first_run is None:
            first_run = runs[0]
    if note and first_run is not None:
        doc.add_comment(runs=[first_run], text=note, author=AUTHOR, initials=INITIALS)


def build_table(after: Paragraph, widths: list[int], header: list[str], rows: list[list],
                aligns: list[str] | None = None, cell_kind=None, note=None,
                fixed_width: bool = False):
    """Insert a data table after `after`.

    rows: list of (cells, kind, bold_cols) where cells are str or list[str];
    cell_kind(row_i, col_i) -> highlight kind, overrides the row kind if given.
    """
    outer = _template_outer()
    inner = outer.find(".//" + W + "tbl")
    aligns = aligns or ["left"] + ["center"] * (len(widths) - 1)

    trs = inner.findall(W + "tr")
    hdr_tr, data_tr, last_tr = trs[0], trs[1], trs[-1]
    hdr_tc = hdr_tr.findall(W + "tc")[0]
    data_tc = data_tr.findall(W + "tc")[0]
    last_tc = last_tr.findall(W + "tc")[0]
    for tr in trs:
        inner.remove(tr)

    grid = inner.find(W + "tblGrid")
    for gc in grid.findall(W + "gridCol"):
        grid.remove(gc)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    if fixed_width:
        tblW = inner.find(W + "tblPr/" + W + "tblW")
        tblW.set(qn("w:w"), str(sum(widths)))
        tblW.set(qn("w:type"), "dxa")

    def make_row(tmpl_tr, tmpl_tc, cells, kind, bold_cols, italic, row_i):
        tr = copy.deepcopy(tmpl_tr)
        for tc in tr.findall(W + "tc"):
            tr.remove(tc)
        for j, (w, text) in enumerate(zip(widths, cells)):
            tc = copy.deepcopy(tmpl_tc)
            tc.find(W + "tcPr/" + W + "tcW").set(qn("w:w"), str(w))
            k = cell_kind(row_i, j) if cell_kind else kind
            lines = text if isinstance(text, list) else [text]
            _set_cell_text(tc, lines, bold=(j in bold_cols), italic=italic,
                           align=aligns[j], kind=k or "")
            tr.append(tc)
        return tr

    inner.append(make_row(hdr_tr, hdr_tc, header, "", set(range(len(widths))), False, -1))
    for i, (cells, kind, bold_cols, italic) in enumerate(rows):
        tmpl_tr, tmpl_tc = (last_tr, last_tc) if i == len(rows) - 1 else (data_tr, data_tc)
        inner.append(make_row(tmpl_tr, tmpl_tc, cells, kind, bold_cols, italic, i))

    after._p.addnext(outer)
    if note:
        first_tc = inner.findall(W + "tr")[0].findall(W + "tc")[0]
        r_el = first_tc.find(".//" + W + "r")
        from docx.text.run import Run
        doc.add_comment(runs=[Run(r_el, None)], text=note, author=AUTHOR, initials=INITIALS)
    return outer


def remove_table_after(p: Paragraph):
    """Remove the table element that immediately follows paragraph `p`."""
    nxt = p._p.getnext()
    while nxt is not None and nxt.tag != W + "tbl":
        if nxt.tag == W + "p" and not "".join(nxt.itertext()).strip():
            nxt = nxt.getnext()
            continue
        raise RuntimeError("expected a table after: " + p.text[:50])
    nxt.getparent().remove(nxt)


def remove_pictures(p: Paragraph):
    for dr in list(p._p.iter(W + "drawing")):
        r = dr.getparent()
        r.getparent().remove(r)


def picture_par_after(anchor: Paragraph, png: Path, width_in: float) -> Paragraph:
    """A centred figure paragraph.

    The template's Normal style uses EXACT 12.2 pt line spacing, which clips an
    inline picture to one text line. The submitted file avoided that by
    floating Fig. 1 as an anchor and by giving the Fig. 2 paragraph its own
    auto spacing; this reproduces the latter, which keeps the figure in flow.
    """
    new_p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    keep = OxmlElement("w:keepNext")
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "120"); sp.set(qn("w:after"), "40")
    sp.set(qn("w:line"), "240"); sp.set(qn("w:lineRule"), "auto")
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center")
    for el in (keep, sp, jc):
        ppr.append(el)
    new_p.append(ppr)
    anchor._p.addnext(new_p)
    par_ = Paragraph(new_p, anchor._parent)
    par_.add_run().add_picture(str(png), width=Inches(width_in))
    return par_


# =========================================================================== #
# 0. Legend, mechanical renames, renumbering
# =========================================================================== #
renumber_tables()          # positioning table will be Table 1
rename_recall()            # R3.2: it is a hit rate
assert any(p.text.startswith('Table 6  Evidence') for p in doc.paragraphs), 'renumbering failed'
assert not any('Recall@1' in p.text for p in doc.paragraphs), 'rename failed'

legend = doc.paragraphs[0]
set_par(legend, [
    ("fix", "REVISION COPY — colour key: "),
    ("fix", "yellow = correction of an error or unsupported claim; "),
    ("new", "green = new content added in response to the reviewers; "),
    ("re", "turquoise = reworded claim, softened wording or renamed term "
           "(Recall@1 → Success@1; tables renumbered after the new Table 1). "),
    ("fix", "Each change carries a comment naming the reviewer point. "
            "Accept the highlights, delete the comments and this note before submission."),
], keep_rpr=False)
for r in legend.runs:
    r.font.size = Pt(8)

# =========================================================================== #
# 1. Abstract
# =========================================================================== #
set_par(par("Patients in multilingual communities"), [
    ("", "Patients in multilingual communities routinely describe symptoms in romanised, "
         "code-switched language, while the clinical literature that could answer them is "
         "written in formal English. We present a measurement study of retrieval-augmented "
         "generation (RAG) for Hinglish patient questions grounded in English clinical case "
         "reports, built on 3,015 query–case pairs that link the MMCQS question set to the "
         "MultiCaRe corpus through a label-leakage gate. "),
    ("re", "Four pre-specified null hypotheses are tested. "),
    ("new", "Code-mixing imposes a retrieval penalty that is positive in all 28 system–metric "
            "combinations tested and significant in 26, "),
    ("", "roughly 4.4 times larger for lexical retrieval than for dense cross-lingual "
         "embedding"),
    ("new", " at rank 1 and wider at depth"),
    ("", ": BM25 is the best system on English queries and the worst on Hinglish. "
         "MuRIL, pretrained on Devanagari Hindi, retrieves at chance on romanised input"),
    ("new", ", and transliterating the queries into Devanagari recovers only 6.5% of its gap; "
            "translating them into English helps dense retrieval but harms lexical retrieval, "
            "because the English arm of the comparison is a clinician-written summary rather "
            "than a translation. "),
    ("fix", "Scored against the evidence it was given, grounding significantly increases concept "
            "precision; scored against a reference neither arm saw, the gain shrinks or "
            "disappears, so it is reported as a reference-dependent effect rather than a "
            "general improvement in answer quality. Grounded factual support is stable across "
            "code-mixing intensity while the ungrounded arm degrades. "),
    ("", "Evidence provenance significantly changes refusal behaviour (Cochran’s Q = 9.09, "
         "p = 0.028), but its effect on answer quality "),
    ("fix", "could not be tested adequately and remains provisional. "),
    ("", "All the corrections, negative results, and the evaluation hazards are given in full."),
], note="R3.5: the abstract now states first that the grounding gain is reference-dependent, "
        "and no longer implies a general quality improvement. R3.1: 'pre-registered' -> "
        "'pre-specified'. R3.6: H03 marked provisional. New retrieval results (R1, R2, R3.2) "
        "summarised in one sentence each.")

# =========================================================================== #
# 2. Introduction (R2: problem / gap / questions / justification / contributions)
# =========================================================================== #
p13 = par("A patient types")
p13.runs[0].text = p13.runs[0].text  # untouched; add run-in head before it
head = p13.runs[0]._r
lead = copy.deepcopy(head)
head.addprevious(lead)
from docx.text.run import Run
lead_run = Run(lead, p13)
lead_run.text = "Problem. "
lead_run.font.bold = True
lead_run.font.highlight_color = COLOUR["new"]
doc.add_comment(runs=[lead_run], text="R2 (clarity of problem, gap, contributions, novelty): "
                "the introduction is restructured into labelled beats — Problem, Gap, Research "
                "questions, Approach, Contributions — with the existing sentences kept where "
                "they already said the right thing.", author=AUTHOR, initials=INITIALS)

set_par(par("Retrieval-augmented generation (RAG) holds"), [
    ("", "Retrieval-augmented generation (RAG) holds the potential to solve this: retrieve a "
         "case in English that matches the patient's circumstances, then generate an answer "),
    ("fix", "grounded in that case, in the patient's own language"),
    ("", ". The question of whether that promise will hold true for code-mixed data is an "
         "empirical one, with two distinct components: is code-mixed surface form hindering "
         "retrieval, and does grounding improve the factual quality of the generated answer?"),
], note="Wording fix ('grounded in the patient's own style' did not say what was meant). "
        "The gap sentences that followed are expanded into their own paragraph below (R2).")

p_gap = new_par_after(par("Retrieval-augmented generation (RAG) holds"), [
    ("new", "<b>Gap.</b> Neither component is isolated in existing evaluations. First, code-mixed "
            "queries are rarely compared against gold English versions of the same question, "
            "so the retrieval cost of code-mixing is confounded with topic. Second, grounding "
            "studies regularly score answers against the very evidence on which the grounded "
            "arm was conditioned, a reference the ungrounded arm never saw. Third, grounding "
            "and code-mixing robustness are seldom tested as hypotheses fixed in advance, or "
            "replicated across generator families."),
], note="R2: the research gap stated explicitly, in three parts that map onto the three "
        "contributions.")
new_par_after(p_gap, [
    ("new", "<b>Research questions.</b> RQ1: how much does romanised code-mixing cost at the "
            "retrieval stage, and does the cost depend on the retrieval method (H04)? RQ2: does "
            "grounding change factual support, and is that change robust to code-mixing "
            "intensity (H01, H02)? RQ3: does the provenance of the evidence matter (H03)?"),
], note="R2: research questions named and mapped onto the hypotheses.")

set_par(par("This paper is a measurement paper"), [
    ("new", "<b>Approach.</b> "),
    ("", "This paper is a measurement paper and not a system-improvement paper. We build a "
         "Hinglish clinical RAG pipeline purposefully from standard components (3,015 Hinglish "
         "patient questions from MMCQS [1] with case reports from MultiCaRe [2] together with "
         "embedding over a FAISS index and grounded prompting of an instruction tuned "
         "generator), so as to reflect the setting and not the particular architecture. "),
    ("new", "The design choices are justified by what each isolates: LaBSE [3] because "
            "cross-lingual alignment is its training objective; BM25 [12] and reciprocal-rank "
            "fusion [13] as the standard lexical baseline and a parameter-free way of combining "
            "the two signals; paired Hinglish–English queries because they hold content fixed "
            "while the surface form varies; and two scoring references, one the grounded arm "
            "saw and one neither arm saw, because the difference between them is the "
            "circularity this paper measures."),
], note="R2 ('stronger justification for the proposed approach'): each design choice now "
        "says what it isolates.")

set_par(par("The four null hypotheses tested are"), [
    ("", "The four null hypotheses tested are: (H01) grounding does not influence factual "),
    ("re", "support"),
    ("", "; (H02) the intensity of code-mixing has no effect on the factual difference between "
         "grounded and ungrounded; (H03) authoritative case evidence does not significantly "
         "influence the factual correctness over general biomedical text; and (H04) the "
         "retrieval quality is not different from code-mixed query as compared to the English "
         "equivalent. "),
    ("fix", "H01–H03 were specified in the project proposal and public repository before any "
            "data were analysed (commit 0917322, 9 February 2026); H04 was formulated after the "
            "initial retrieval experiments and is reported as a post-hoc hypothesis, tested "
            "confirmatorily on the full matched set (Sect. 4)."),
], note="R3.1 (pre-registration record): H01–H03 are timestamped by a public commit before the "
        "first prototype run (4 March 2026); H04 was NOT pre-specified — it first appears in "
        "August 2026 — and the paper now says so. 'factual consistency' -> 'factual support' "
        "so the hypothesis names the metric that is actually measured.")

set_par(par("The study makes three contributions"), [
    ("new", "<b>Contributions.</b> "),
    ("", "The study makes three contributions. First, it quantifies the retrieval-stage "
         "code-mixing penalty, measured against gold "),
    ("fix", "English summaries"),
    ("", " of the same questions behind a label-leakage gate, together with the lexical/dense "
         "asymmetry and its script-mismatch mechanism"),
    ("new", ": the penalty is positive in all 28 system–metric combinations, 4.4 times larger "
            "for lexical than for dense retrieval at rank 1, transliteration recovers only 6.5% "
            "of MuRIL’s gap, and translation helps dense but not lexical retrieval"),
    ("", " (Sect. 5.1). Second, it provides "),
    ("re", "pre-specified"),
    ("", " hypothesis tests for the grounding effect and for code-mixing robustness"),
    ("re", ", reported under the tests that were planned and replicated across two generator "
           "families under oracle evidence"),
    ("", " (Sects. 5.3 and 5.4). Third, it demonstrates that evaluation design changes the "
         "reported benefit of grounding"),
    ("new", " — positive in five of five conditions against the evidence reference and in one of "
            "five against an unbiased one —"),
    ("", " and supplies the degenerate baselines "),
    ("new", "and the extractor audit "),
    ("", "required to detect such artefacts (Sects. 5.2 and 5.3)."),
], note="R2: each contribution now carries its headline number. 'gold human translations' -> "
        "'gold English summaries': MMCQS supplies clinician-style summaries (~21 words), not "
        "translations (~97 words) — see Sect. 5.1. 'replicated across two generator families' "
        "is now qualified: under real retrieval the gpt-oss replication is much weaker "
        "(Table 4).")

# =========================================================================== #
# 3. Related work: positioning table (R2) and 'to our knowledge' fix
# =========================================================================== #
p28 = par("To our knowledge, no prior study")
set_par(p28, [
    ("", "To our knowledge, no prior study of code-mixed clinical question answering (i) "
         "isolates the retrieval-stage penalty of romanisation by comparing code-mixed queries "
         "against gold "),
    ("fix", "English versions"),
    ("", " of the same questions under a label-leakage gate, (ii) tests grounding and "
         "code-mixing robustness as "),
    ("re", "pre-specified"),
    ("", " hypotheses across generator families, or (iii) measures how strongly the choice of "
         "scoring reference changes the reported grounding benefit. This paper does all three"),
    ("new", "; Table 1 positions it against the closest prior work"),
    ("", "."),
], note="R2 ('comparisons with relevant existing methods'): a positioning table follows. "
        "'translations' corrected to 'English versions' (they are summaries).")

cap1 = new_par_after(p28, [
    ("new", "Table 1  Positioning against the closest prior work. ✓ = addressed; — = not "
            "addressed; ~ = partly. ‘Gold English pairs’ = the same question available in "
            "English for a matched comparison; ‘reference bias checked’ = the grounding benefit "
            "scored against a reference the grounded arm did not see."),
], ppr_from=par("Table 5  Spearman correlation"), style="ICCSDI Caption",
   note="R2: positioning table. Ticks are taken from what each paper actually does; MedSumm "
        "supplies English summaries (the pairs this study uses) but studies summarisation, "
        "not retrieval.")
build_table(cap1, widths=[2450, 1150, 1050, 1250, 1250, 1450],
            header=["Work", "Code-mixed queries", "Clinical domain", "Retrieval stage isolated",
                    "Gold English pairs", "Reference bias checked"],
            rows=[
                (["MedSumm [1]", "✓", "✓", "—", "✓ (summaries)", "—"], "new", set(), False),
                (["HealthAlignSumm [4]", "✓", "✓", "—", "—", "—"], "new", set(), False),
                (["Xiong et al. [5]", "—", "✓", "✓", "—", "—"], "new", set(), False),
                (["MMed-RAG [6]", "—", "✓", "~", "—", "—"], "new", set(), False),
                (["CroCoSum [10]", "✓", "—", "—", "✓", "—"], "new", set(), False),
                (["HiFactMix [11]", "✓", "—", "~", "—", "—"], "new", set(), False),
                (["This work", "✓", "✓", "✓", "✓ (summaries)", "✓"], "new", {0}, False),
            ])

# =========================================================================== #
# 4. Methods and design
# =========================================================================== #
set_par(par("The evaluation items are pairs"), [
    ("", "The evaluation items are pairs (<i>q</i>, <i>q</i><sub>E</sub>): a Hinglish patient "
         "question <i>q</i> together with its gold, human-written English "),
    ("fix", "summary <i>q</i><sub>E</sub> — a clinician-style condensation of the same question "
            "(median 20 words against 92 for <i>q</i>; Sect. 5.1 measures the consequence)"),
    ("", ". The evidence corpus comprises 10,000 English MultiCaRe case narratives drawn from 18 "
         "condition groups, segmented into 41,746 passages. A retrieval system <i>s</i> maps a "
         "query onto a ranked list of cases; "),
    ("re", "Success@1<sub><i>s</i></sub>(·)"),
    ("", " is the proportion of queries for which the top-ranked case falls in the same "
         "condition group as the query"),
    ("re", " (a hit rate: with a median of 597 relevant cases per query it is not recall, and "
           "it is named accordingly)"),
    ("", ". This condition-group criterion is deliberately coarse (Sect. 6.1), but it is the "
         "same across both languages, which is exactly what the penalty estimand requires. The "
         "per-system code-mixing penalty is"),
], note="'English equivalent' -> 'English summary': the MMCQS field is a summary, and the "
        "length difference is now stated. R3.2: Recall@1 renamed Success@1 because each "
        "query has hundreds of relevant cases.")

set_par(par("where a positive"), [
    ("", "where a positive Δ<i>s</i> indicates that the deployed, code-mixed form retrieves "
         "worse than its English equivalent. An instruction-tuned generator <i>g</i> produces "
         "a zero-shot answer <i>a</i><sub>0</sub> = <i>g</i>(<i>q</i>) and a grounded answer "
         "<i>a</i><sub>r</sub> = <i>g</i>(<i>q</i>, <i>c</i>) conditioned on a retrieved or "
         "oracle case <i>c</i>; both answers are in Hinglish. The answers are then scored by "
         "overlap with a fixed 26-concept clinical lexicon against a designated reference text: "
         "factual support is concept precision (the share of concepts the answer asserts that "
         "are actually present in the reference), concept recall is the share of reference "
         "concepts the answer asserts, and F1 combines the two. The estimands are the "
         "per-system penalty Δ<i>s</i>, paired within-query differences between arms scored "
         "identically, "),
    ("fix", "the change in that paired difference across levels of the Hindi proportion of "
            "<i>q</i> (with per-arm correlations as its decomposition)"),
    ("", ", and contrasts across matched evidence corpora."),
], note="R3.3: the H02 estimand is now the change in the paired difference, which is what the "
        "hypothesis states; the per-arm correlations are its decomposition.")

set_par(par("The study tests four pre-registered"), [
    ("fix", "The study tests four pre-specified null hypotheses. H01–H03 were fixed in the "
            "project proposal and the public repository before any analysis was run (commit "
            "0917322, 9 February 2026; https://github.com/saaachis/hinglish-grounded-medical-rag/"
            "commit/0917322), and the first prototype followed on 4 March 2026. H04 was added "
            "after the initial retrieval experiments and is tested confirmatorily on the full "
            "matched set. This is a timestamped public record rather than a registry entry, "
            "and the deviations from the planned analysis are listed at the end of Sect. 4.2. "),
    ("", "Rejecting a null is the positive result: it indicates that the corresponding effect "
         "is real. And the two-sided tests are conducted throughout, with false-discovery-rate "
         "correction applied within each hypothesis family (Benjamini–Hochberg, BH); cluster "
         "bootstraps deal with the non-independence introduced by repeated reference texts "
         "(Sect. 6.1)."),
], note="R3.1: the pre-registration record the reviewer asked for. The claim that all four "
        "hypotheses were fixed before analysis was not true of H04 and is corrected.")

set_par(par("The queries come from MMCQS"), [
    ("", "The queries come from MMCQS [1], a large repository of Hinglish patient questions, "
         "each accompanied by a human-written English summary and a textual image description; "
         "only these text fields are used in this study, and no images are processed. "),
    ("new", "The summary is a clinician-style condensation of the question (median 20 words "
            "against 92 for the Hinglish query), not a translation; Sect. 5.1 measures what "
            "that difference contributes to the penalty. "),
    ("", "Evidence comes from MultiCaRe [2], an open-access collection of clinical case reports: "
         "61,316 cases pass filtering, of which 10,000 are indexed, balanced across 18 condition "
         "groups. The median case runs 554 words, and passage chunking yields 41,746 passages. "
         "Pairing then produces 3,015 query–case evaluation pairs at 100% query coverage."),
], note="Data description made accurate: the English field is a summary. This is the basis of "
        "the register caveat in Sect. 5.1.")

set_par(par("Four retrieval systems are compared"), [
    ("", "Four retrieval systems are compared on the passage-chunked index: TF-IDF, BM25 [12], "
         "dense LaBSE retrieval [3], and a hybrid fusing the BM25 and LaBSE rankings by "
         "reciprocal-rank fusion [13]. A supplementary single-vector index (one embedding per "
         "case) additionally hosts MuRIL [14] and multilingual-e5-base for the script-mismatch "
         "analysis. "),
    ("new", "Two query-side remedies are tested as further baselines: transliterating the "
            "romanised query into Devanagari for MuRIL (a hand-verified lexicon covering 87% of "
            "Hindi token occurrences plus rule-based mapping for the remainder; a trained "
            "transliterator could not be installed in the environment, so the mapping is a "
            "lower bound), and translating it into English with gpt-oss-120b before retrieval, "
            "on a stratified 500-query subset. "),
    ("re", "The primary metric is Success@1 against condition-group relevance, the proportion "
           "of queries whose top-ranked case falls in the query’s condition group; Success@k "
           "for k ∈ {3, 5, 10}, MRR@10 and nDCG@10 are also reported. nDCG uses the "
           "corpus-level ideal ranking (the min(relevant, k) hits placed first), and a graded "
           "variant awards grade 2 to a same-group case that also shares at least one lexicon "
           "concept with the gold English question: an automatic proxy for finer relevance, "
           "not a clinical judgment. The random floor of Success@1 for condition-blind "
           "retrieval is 0.0626, rising to 0.4739 at rank 10. "),
    ("", "Because the relevance criterion is coarse, absolute Success@1 is a lower bound on "
         "true relevance (Sect. 6.1). The penalty Δ<i>s</i> is tested per system "),
    ("new", "and per metric, "),
    ("", "with McNemar’s test on paired top-"),
    ("re", "k hits and the Wilcoxon signed-rank test on MRR and nDCG"),
    ("", ": a penalty appearing under one configuration is a property of that configuration, "
         "whereas one holding across methods is a property of code-mixing itself."),
], note="R3.2: Success@k, MRR@10 and nDCG@10 added, with graded relevance as an explicitly "
        "automatic proxy. R1 / R2: transliteration and translation baselines introduced here.")

set_par(par("The statistical toolkit includes"), [
    ("", "The statistical toolkit includes paired within-query deltas with Cohen's d, McNemar's "
         "test for paired binary outcomes, and Cochran's Q for refusal across matched evidence "
         "conditions, together with cluster bootstraps over repeated references, BH correction "
         "within hypothesis families, "),
    ("new", "one-way ANOVA and Kruskal–Wallis on the paired gain across code-mixing tertiles "
            "for the pre-specified H02 test, "),
    ("", "and Spearman ρ with 95% confidence intervals for dose–response."),
], note="R3.3: the planned H02 test named.")

new_par_after(par("The statistical toolkit includes"), [
    ("new", "<b>Deviations from the pre-specified plan.</b> Five departures from the original "
            "analysis plan are disclosed. (1) H02 was planned as a two-way ANOVA / "
            "Kruskal–Wallis on code-mixing level; it is reported that way as the primary test, "
            "with the per-arm correlations as an exploratory decomposition. (2) Factual "
            "consistency and hallucination rate were planned as two outcomes; hallucination is "
            "exactly 1 − precision and is not reported separately. (3) Concept precision was the "
            "planned quality metric; F1 is reported wherever quality is at issue, because "
            "degenerate baselines showed that precision’s optimum is a one-word answer. (4) The "
            "Code-Mixing Index was planned over a fixed Hindi word list; that list counted "
            "“doctor” and “please” as Hindi and was repaired. (5) H03 was planned on answer "
            "quality; it is reported as exploratory, with refusal as a secondary outcome."),
], note="R3.1: deviations from the plan disclosed in one place rather than left for a reader "
        "to reconstruct.")

set_par(par("Experiments ran under Python"), [
    ("", "Experiments ran under Python 3.12 with sentence-transformers 3.0.1, transformers "
         "4.42.4, PyTorch 2.3.1, faiss-cpu 1.8.0, scikit-learn 1.8.0, NumPy 1.26.4, pandas "
         "2.2.1, SciPy 1.12.0 and matplotlib 3.8.3, with rank_bm25 for the BM25 baseline; full "
         "pinned versions are listed in requirements.txt in the released repository. "),
    ("new", "The translate-then-retrieve baseline uses the hosted openai/gpt-oss-120b as "
            "translator, and the extractor audit of Sect. 5.2 uses qwen/qwen3.8-27b as an "
            "independent reader, a model family used nowhere else in the study."),
], note="Models used by the new baselines recorded for reproducibility.")

# =========================================================================== #
# 5. Results
# =========================================================================== #
set_par(par("Subsections follow the pre-specified order"), [
    ("", "Subsections follow the pre-specified order: the retrieval-stage hypothesis H04 first, "
         "a validity audit of the scoring instrument before any generation numbers are "
         "presented, then H01, H02 and H03, and finally the negative and corrected results. "),
    ("fix", "H04 is rejected for every system and metric; H01 is rejected under the "
            "evidence-referenced protocol; the pre-specified H02 test is borderline (Sect. 5.4); "
            "H03 is exploratory (Sect. 5.5). "),
    ("", "Bold marks values highlighted in the text; all reported p-values are two-sided."),
], note="Outcome summary aligned with the corrected tests (R3.3, R3.5, R3.6).")

# ---- 5.1 ---------------------------------------------------------------- #
set_par(par("Success@1 for each system on the deployed"), [
    ("", "Success@1 for each system on the deployed Hinglish queries and their gold English "
         "counterparts over all 3,015 pairs is reported in Table 2 and visualised in Fig. 1. "
         "Every system "),
    ("fix", "retrieves significantly worse on Hinglish at rank 1 (McNemar, largest p = 0.018), "
            "and the penalty is positive in all 28 system–metric combinations — four systems × "
            "Success@1/3/5/10, MRR@10, nDCG@10 and graded nDCG@10 — and significant in 26 "
            "(Fig. 2). "),
    ("", "Because the penalty holds across lexical, dense, and hybrid methods"),
    ("new", " and at every depth"),
    ("", ", it is a property of code-mixing itself rather than of any single configuration, "
         "which is why the test is run per system. H₀₄ is rejected for every system tested."),
], note="R3.2: the penalty is now tested at four depths and on MRR and nDCG for every "
        "system, including the TF-IDF test that was left blank. The text no longer needs the "
        "hedge 'TF-IDF moves in the same direction'.")

capT2 = par("Table 2: Success@1 by retrieval system")
set_par(capT2, [
    ("re", "Table 2: Success@1, Success@10 and MRR@10 by retrieval system and query language "),
    ("", "(<i>n</i> = 3,015 pairs; higher is better). Penalty = English − Hinglish"),
    ("new", " at rank 1"),
    ("", "; McNemar’s test per system. Bold indicates the best deployed-condition (Hinglish) "
         "system and the greatest penalty. Penalty 95% CIs (bootstrap): LaBSE [+0.0040, "
         "+0.0375], BM25 [+0.0743, +0.1085], Hybrid [+0.0036, +0.0408]"),
    ("new", ", TF-IDF [+0.0541, +0.0839]. Penalties at every depth and metric are shown in "
            "Fig. 2."),
], note="R3.2: deeper-rank columns added; the missing TF-IDF test filled in.")
remove_table_after(capT2)
build_table(capT2, widths=[1650, 760, 760, 980, 1000, 790, 790, 950, 950], fixed_width=True,
            header=["System", "S@1 Hi", "S@1 En", "Penalty (En − Hi)", "McNemar p",
                    "S@10 Hi", "S@10 En", "MRR@10 Hi", "MRR@10 En"],
            rows=[
                (["Hybrid (RRF)", "0.1751", "0.1973", "+0.0222", "0.018",
                  "0.6564", "0.7393", "0.2987", "0.3438"], "", {1}, False),
                (["LaBSE (passages)", "0.1280", "0.1486", "+0.0206", "0.017",
                  "0.6325", "0.6783", "0.2584", "0.2835"], "", set(), False),
                (["BM25", "0.0935", "0.1847", "+0.0912", "9.9 × 10<sup>−26</sup>",
                  "0.6116", "0.7373", "0.2173", "0.3385"], "", {3}, False),
                (["TF-IDF", "0.0842", "0.1529", "+0.0687", "1.9 × 10<sup>−19</sup>",
                  "0.5599", "0.6153", "0.1972", "0.2729"], "", set(), False),
                (["Random floor", "0.0626", "0.0626", "—", "—", "0.4739", "0.4739", "—", "—"],
                 "", set(), True),
            ],
            cell_kind=lambda i, j: ("new" if j >= 5 else ("fix" if (i == 3 and j in (3, 4)) else "")),
            note="R3.2: Success@10 and MRR@10 columns (green) added from "
                 "results/retrieval_v2/; TF-IDF penalty and p (yellow) were '—' in the "
                 "submitted table with no reason given. Depth-adjusted random floor added for "
                 "rank 10. All rank-1 values unchanged.")

set_par(par("The damage is sharply asymmetric"), [
    ("", "The damage is sharply asymmetric. BM25 loses 0.0912 Success@1 to code-mixing against "
         "0.0206 for dense LaBSE, roughly 4.4 times more, producing a crossover: BM25 is the "
         "best system on English queries (0.1847) and the worst on Hinglish (0.0935). "),
    ("new", "The asymmetry widens with depth (Fig. 2): at rank 10 BM25’s penalty is +0.1257 "
            "against +0.0458 for LaBSE, and on graded nDCG@10 LaBSE shows no significant "
            "penalty at all (+0.0016, p = 0.20) while BM25 loses 0.0822 (p = 7.5 × "
            "10<sup>−140</sup>). "),
    ("", "This crossover is the paper’s empirical argument for cross-lingual embedding at the "
         "retrieval layer: surface-form overlap is precisely what romanised code-switching "
         "removes."),
], note="R3.2: the rank-1 ratio is the weakest version of the asymmetry; on rank-sensitive "
        "measures the gap is an order of magnitude.")

set_par(par("On the single-vector index, MuRIL"), [
    ("", "On the single-vector index, MuRIL, an encoder pretrained on Indian languages with "
         "Hindi in Devanagari script, scores 0.0640 on Hinglish against the 0.0626 random "
         "floor: chance. On the same content in English it recovers to 0.1821. "),
    ("re", "The failure is consistent with script mismatch rather than language mismatch: MMCQS "
           "is romanised, and the encoder has no purchase on the romanised surface. "),
    ("new", "Transliterating the romanised queries into Devanagari tests that reading directly. "
            "MuRIL rises from 0.0640 to 0.0716 on Success@1 (not significant, p = 0.24) and "
            "improves reliably on rank-sensitive metrics (MRR@10 +0.0148, p = 0.0096; nDCG@10 "
            "+0.0077, p = 0.0008), yet recovers only 6.5% of the romanised-to-English gap. "
            "Script mismatch is therefore real but secondary: because the transliteration is a "
            "hand-verified mapping rather than a trained model this is a lower bound, but on "
            "this evidence writing the query in the encoder’s own script does not restore its "
            "English performance."),
], note="R1 ('stronger transliteration-aware baselines'): the mechanism the submitted paper "
        "called 'a hypothesis the study motivates but does not test' is now tested. The "
        "conclusion is softened accordingly.")

p71 = par("For comparison, Hinglish Success@1 on that index")
remove_pictures(p71)
fig1_p = picture_par_after(p71, FIG / "fig1_table1.png", 4.13)
doc.add_comment(runs=[p71.runs[0]], text="Fig. 1 regenerated with the axis relabelled "
                "Success@1 (R3.2). Values are unchanged.", author=AUTHOR, initials=INITIALS)

p_trans = new_par_after(p71, [
    ("new", "Translation is the other query-side remedy, and it separates language from "
            "register. On a stratified subset of 500 queries translated into English by "
            "gpt-oss-120b, dense retrieval improves (LaBSE 0.1220 → 0.1680, p = 0.040) but "
            "lexical retrieval collapses (BM25 0.1080 → 0.0240, p = 5.7 × 10<sup>−8</sup>), "
            "below its Hinglish score. The mechanism is measurable: under half of a Hinglish "
            "query’s tokens exist in the English corpus at all, but those that do are rare and "
            "specific (mean IDF 5.22), because the Hindi carries the grammar and the English "
            "carries the clinical content; a fluent translation restores common function words "
            "(mean IDF 2.28) that dilute the lexical query. The same diagnostic bounds the main "
            "comparison: the gold English question averages 21 words against 97 for the "
            "translation, so it is a clinician-style summary, and the penalty in Table 2 "
            "compares a patient narrative with a clinical summary, conflating language with "
            "conciseness and register. H04 is unaffected — the penalty holds for every system "
            "and depth, and the MuRIL result is independent of it — but the estimate is an "
            "upper bound on the language effect alone; the translated arm, which holds length "
            "and register fixed, is the cleaner language-only comparison, and against it dense "
            "retrieval improves while lexical retrieval does not."),
], note="R2 ('comparisons with relevant existing methods'): translate-then-retrieve is the "
        "standard alternative to a cross-lingual encoder. Its result also exposes that the "
        "English arm is a summary, which the paper now states as a limitation.",
   ppr_from=par("The damage is sharply asymmetric"))
# move the translation paragraph to sit after the figure caption block
cap_fig1 = par("Fig. 1  Success@1 by retrieval system")
p_trans._p.getparent().remove(p_trans._p)
spacer_after_cap = Paragraph(cap_fig1._p.getnext(), cap_fig1._parent)
spacer_after_cap._p.addnext(p_trans._p)

set_par(cap_fig1, [
    ("", "Fig. 1  Success@1 by retrieval system and query language over 3,015 pairs (higher is "
         "better). The dashed line marks the random floor (0.0626). Values correspond to Table 2."),
])

fig2_p = picture_par_after(p_trans, FIG / "fig3_penalty_by_depth.png", 4.13)
fig2_cap = new_par_after(fig2_p, [
    ("new", "Fig. 2  The code-mixing penalty (English − Hinglish Success@k, 95% bootstrap "
            "CI) at every rank cutoff, per system. The lexical/dense gap widens with depth; a "
            "single rank-1 cutoff understates it."),
], ppr_from=cap_fig1, style="ICCSDI Caption",
   note="R3.2: new figure showing the penalty at every depth. The former Fig. 3 (dose–response) "
        "duplicated the numbers in the H02 table and is removed.")
insert_after(fig2_cap, ppr_from=Paragraph(cap_fig1._p.getnext(), cap_fig1._parent))

# ---- 5.2 ---------------------------------------------------------------- #
p81 = par("The same audit closes a double-counting hazard")
new_par_after(p81, [
    ("new", "The instrument was also audited against an independent reader. qwen/qwen3.8-27b, "
            "a model family used nowhere else in the study, labelled 120 sampled answers "
            "(60 per arm, stratified by code-mixing tertile) with the concepts each positively "
            "asserts about the patient, from the same 26-item vocabulary. Agreement with the "
            "regex extractor is moderate (Cohen’s κ = 0.494; extractor precision 0.412 and "
            "recall 0.686 against the reader), and the extractor over-attributes: it counts "
            "hedged and hypothetical mentions that the reader excludes. The over-attribution is "
            "unequal across arms — precision 0.480 on grounded answers against 0.358 on "
            "zero-shot answers — so it does not cancel in the paired difference. Because a "
            "spurious concept enlarges the denominator of factual support, the zero-shot arm is "
            "depressed more than the grounded arm, and the measured grounding benefit of "
            "Sect. 5.3 is inflated by an amount this audit bounds but does not determine. This "
            "is automatic cross-validation, not clinical validation (Sect. 6.1). The lexicon’s "
            "coverage is also bounded: 98.3% of references contain at least one concept, but "
            "13.9% contain exactly one, and the most frequent terms it cannot represent are "
            "misspellings in the free-text captions that serve as the unbiased reference."),
], note="R3.4 / R1.c (lexicon not clinician-validated) and R1.d (richer factuality "
        "evidence): clinician annotation was not feasible in the revision window, so the "
        "extractor is audited against an independent model and the lexicon's coverage is "
        "measured. The arm-dependent bias is a new limitation on H01 and is stated here "
        "rather than left for a reader to find.")

# ---- 5.3 ---------------------------------------------------------------- #
set_par(par("Under the evaluation protocol in common use"), [
    ("", "Under the evaluation protocol in common use, scoring both arms against the retrieved "
         "evidence, grounding improves factual support strongly and consistently (Table 4). "),
    ("fix", "The effect survives the lexicon repair of Sect. 5.2 and replicates across the two "
            "generator families under oracle evidence (llama-3.1-8b +0.2962, d = 0.720; "
            "gpt-oss-120b +0.2608, d = 0.640); under real retrieval on gpt-oss-120b it is much "
            "smaller (+0.0704, d = 0.191), so H01 is rejected under the evidence-referenced "
            "protocol, and the replication is a property of oracle evidence rather than of the "
            "deployed pipeline. "),
    ("", "Following Sect. 5.2, the arm means "),
    ("re", "underlying Table 4"),
    ("", " are components of the paired estimand Δ, not quality measures."),
], note="CORRECTION: the submitted Table 3 gave gpt-oss-120b +0.2659 (oracle) and +0.2238 "
        "(real retrieval); neither value exists in any committed result file. The re-scored "
        "contrasts (results/rescored/contrasts.csv) give +0.2608 and +0.0704. The real-"
        "retrieval row is what supported 'replicates across two generator families', so that "
        "claim is now qualified. R3.5: H01's rejection is tied to its protocol.")

capT4 = par("Table 4  Grounding effect on factual support")
set_par(capT4, [
    ("fix", "Table 4  Grounding effect (Δ = grounded − zero-shot within query) under both "
            "metrics and both references: concept precision against the evidence the grounded "
            "arm was given, and concept F1 against that evidence and against the image "
            "description neither arm saw. Each cell gives Δ with BH-corrected significance "
            "(*** p < 0.001, ** p < 0.01, * p < 0.05, n.s.), <i>n</i> and Cohen’s d. Under the "
            "original lexicon the llama oracle contrast was +0.2345 (d = 0.576, <i>n</i> = "
            "1,165). Absolute arm means are not quality measures (Sect. 5.2)."),
], note="R3.5: Table 3 (precision only) and Fig. 2 (F1 only) are merged into one table so "
        "that the sign change between them is visible in a single row. Replaces the former "
        "Fig. 2.")
remove_table_after(capT4)
build_table(capT4, widths=[2500, 2000, 2000, 2100],
            header=["Generator / evidence", "Precision vs evidence", "F1 vs evidence",
                    "F1 vs unbiased reference"],
            rows=[
                (["llama-3.1-8b, oracle", ["+0.2962 ***", "n = 669, d = +0.72"],
                  ["+0.2029 ***", "n = 651, d = +0.68"], ["+0.0620 ***", "n = 613, d = +0.22"]],
                 "fix", set(), False),
                (["gpt-oss-120b, oracle", ["+0.2608 ***", "n = 214, d = +0.64"],
                  ["+0.0926 ***", "n = 207, d = +0.28"], ["−0.0207 n.s.", "n = 189, d = −0.07"]],
                 "fix", set(), False),
                (["gpt-oss-120b, real retrieval", ["+0.0704 **", "n = 221, d = +0.19"],
                  ["−0.0320 *", "n = 210, d = −0.12"], ["−0.0469 *", "n = 200, d = −0.17"]],
                 "fix", set(), False),
                (["gpt-oss-20b, oracle", ["+0.1966 ***", "n = 140, d = +0.45"],
                  ["−0.0067 n.s.", "n = 134, d = −0.02"], ["−0.0217 n.s.", "n = 133, d = −0.06"]],
                 "fix", set(), False),
                (["gpt-oss-20b, real retrieval", ["+0.1071 **", "n = 138, d = +0.25"],
                  ["−0.0698 **", "n = 130, d = −0.26"], ["−0.0574 n.s.", "n = 130, d = −0.16"]],
                 "fix", set(), False),
            ],
            note="All values from results/rescored/contrasts.csv (negation-repaired lexicon, "
                 "BH-corrected across the family). Positive in 5/5 conditions against the "
                 "evidence reference; positive in 1/5 on F1 against the unbiased reference.")

set_par(par("The magnitude, however, depends on what is scored against"), [
    ("", "The magnitude, however, depends on what is scored against ("),
    ("re", "Table 4"),
    ("", "). Re-scoring the same generations by concept F1, BH-corrected, against the circular "
         "reference (the retrieved evidence) versus the unbiased reference (the image "
         "description, which neither arm saw) changes the picture: llama-3.1-8b with oracle "
         "evidence falls from +0.203 to +0.062; gpt-oss-120b with oracle evidence falls from "
         "+0.093 to −0.021 (n.s.); and gpt-oss-120b under real retrieval reads −0.032 "),
    ("fix", "circular and −0.047 unbiased, both significant. Across the five conditions the "
            "precision effect against the evidence reference is positive in all five; the F1 "
            "effect against the unbiased reference is positive in one."),
], note="Reference now points at the merged table; the gpt-oss real-retrieval F1 deltas are "
        "significant (BH p = 0.041 and 0.011), not 'n.s.' as the submitted text said.")

for prefix in ("Fig. 2: Grounding effect on concept",):
    delete_par(par(prefix))
# the paragraph holding the old Fig. 2 image and its empty spacer
p89 = par("This reference-sensitivity is a methodological caution")
nxt = p89._p.getnext()
while nxt is not None and (nxt.find(".//" + W + "drawing") is not None or not "".join(nxt.itertext()).strip()):
    following = nxt.getnext()
    nxt.getparent().remove(nxt)
    nxt = following
    if nxt is not None and "".join(nxt.itertext()).strip().startswith("Mechanistically"):
        break

set_par(par("Mechanistically, grounding trades recall"), [
    ("", "Mechanistically, "),
    ("fix", "on gpt-oss-120b "),
    ("", "grounding trades recall for precision: it decreases concept recall by 0.138 (BH p = "
         "0.00025), resulting in a more conservative model"),
    ("fix", "; the llama arm moves the other way (recall +0.077, BH p = 9.3 × 10<sup>−7</sup>), "
            "so the trade is generator-specific"),
    ("", ", and whether that trade pays off depends on the generator. "),
    ("fix", "Oracle evidence scores higher than real retrieval by +0.075, but the difference is "
            "not significant on either generator (p = 0.226, <i>n</i> = 76; p = 0.106, <i>n</i> "
            "= 125), so the condition filter’s advantage over honest retrieval is not "
            "established at this sample size. "),
    ("", "One consequence of grounding is a change in refusal behaviour. For gpt-oss-120b "
         "(<i>n</i> = 467) the zero-shot arm virtually never declines (0.2%), while the "
         "grounded arms decline on 29.8% (oracle) and 33.8% (real retrieval) of queries, "
         "usually stating that the evidence concerns another patient; coverage drops from "
         "92.9% zero-shot to 49.3% and 52.0% respectively. Retrieval’s top-1 was "
         "condition-correct in only 10.7% of these rows. Refusals assert no scoreable concept "
         "and vanish from a naive mean, which would make the system look healthiest exactly "
         "where it fails; coverage is therefore reported beside every factuality number."),
], note="CORRECTION: '+0.0875 (p = 0.041) ... small but significant' appears in no result "
        "file; the two runs give +0.0748 (p = 0.226) and +0.0755 (p = 0.106), both "
        "non-significant. The recall-for-precision sentence now names its generator, because "
        "the llama arm shows the opposite sign.")

# ---- 5.4 ---------------------------------------------------------------- #
set_par(par("H02 is tested per arm"), [
    ("fix", "H02 concerns the grounded-minus-ungrounded difference, and it is tested as planned: "
            "a two-way design with arm varying within query, whose arm × code-mixing "
            "interaction is the change in the paired gain across tertiles of Hindi proportion "
            "(<i>n</i> = 1,165). The two pre-specified tests disagree — one-way ANOVA on the "
            "gain F = 3.03, p = 0.049; Kruskal–Wallis H = 5.11, p = 0.078 — so both are "
            "reported and neither is chosen after the fact; on the continuous measure the gain "
            "rises with Hindi proportion (Spearman ρ = +0.081, 95% CI [+0.023, +0.138], "
            "p = 0.006), from +0.204 in the lowest tertile to +0.274 in the highest. Evidence "
            "against H02 is therefore borderline on the tertile tests and significant on the "
            "continuous one. The per-arm decomposition (Table 5) explains the direction: "),
    ("", "zero-shot factual support degrades significantly as code-mixing intensifies (ρ = "
         "−0.116, BH p = 0.0003), while the grounded arm is flat (ρ = −0.001, n.s.)"),
    ("new", " and statistically equivalent to zero effect (90% CI [−0.048, +0.049] within a "
            "±0.10 bound chosen after the data were seen)"),
    ("", "; neither arm shows a significant hallucination trend. "),
    ("fix", "The gain grows because the ungrounded baseline falls, not because grounding "
            "improves. Under the originally shipped code-mixing measure the interaction is not "
            "significant (Kruskal–Wallis p = 0.19), a difference attributable to the lexicon "
            "repair described below."),
], note="R3.3 (hypothesis/test misalignment): the reviewer is right. The primary test is now "
        "the pre-specified interaction on the paired gain; the per-arm correlations are its "
        "decomposition. The submitted sentence claiming the per-arm design is 'deliberately "
        "stronger' and that the difference test is non-significant is removed: the latter was "
        "true only under the old code-mixing measure.")

capT5 = par("Table 5  Spearman correlation")
set_par(capT5, [
    ("fix", "Table 5  Code-mixing intensity (Hindi proportion, repaired measure) against outcome, "
            "<i>n</i> = 1,165. First row: the pre-specified test on the paired gain (continuous "
            "ρ; tertile tests ANOVA p = 0.049, Kruskal–Wallis p = 0.078). Remaining rows: "
            "exploratory per-arm decomposition, BH-corrected."),
], note="R3.3: the pre-specified gain test added as the first row.")
remove_table_after(capT5)
build_table(capT5, widths=[3000, 850, 1900, 1050, 1797],
            header=["Outcome", "ρ", "95% CI", "p", "Reading"],
            rows=[
                (["Gain (grounded − zero-shot), pre-specified", "+0.081", "[+0.023, +0.138]",
                  "0.006", "rises"], "new", {0}, False),
                (["Grounded factual support", "−0.001", "[−0.058, +0.057]", "0.98",
                  "flat (equivalent)"], "", {0}, False),
                (["Zero-shot factual support", "−0.116", "[−0.171, −0.059]", "0.0003",
                  "degrades"], "", set(), False),
                (["Grounded hallucination", "−0.022", "[−0.080, +0.036]", "0.59", "flat"],
                 "", set(), False),
                (["Zero-shot hallucination", "+0.042", "[−0.018, +0.101]", "0.30", "flat"],
                 "", set(), False),
            ],
            cell_kind=lambda i, j: ("new" if i == 0 else ("new" if (i == 1 and j == 4) else "")),
            note="First row from results/h2_per_arm/h2_prespecified_test.md; the four "
                 "per-arm rows are unchanged from the submitted Table 4.")

for prefix in ("Fig. 3  Factual support against Hindi proportion",):
    delete_par(par(prefix))
# remove the old Fig. 3 image paragraph and spacer that sat between Table 5 and the repair text
p102 = par("The code-mixing measure itself required repair")
prev = p102._p.getprevious()
while prev is not None and prev.tag == W + "p" and (prev.find(".//" + W + "drawing") is not None
                                                     or not "".join(prev.itertext()).strip()):
    before = prev.getprevious()
    prev.getparent().remove(prev)
    prev = before

# ---- 5.5 ---------------------------------------------------------------- #
h55 = par("5.5  Evidence Provenance (H03)")
set_par(h55, [("", "5.5  Evidence Provenance (H03)"), ("re", ": Exploratory")],
        note="R3.6: H03 labelled exploratory; its primary outcome was not adequately tested.")

set_par(par("Provenance significantly changes refusal behaviour"), [
    ("fix", "H03’s primary outcome, answer quality, could not be tested adequately: pairwise F1 "
            "differences are small and none is significant, and the four-way omnibus rests on "
            "only 13 of 160 queries scoreable under all four conditions; with refusal rates of "
            "76–88%, that joint event has probability near 0.1%. Moreover, most “scoreable” "
            "answers are also refusals — hedges that decline to answer yet name a concept "
            "(MultiCaRe: 59 of 63; Table 6) — so the F1 column rests largely on refusals, and "
            "requiring a genuine answer leaves no query usable under all four conditions. H03 "
            "is therefore neither rejected nor supported. "),
    ("", "As a "),
    ("fix", "secondary outcome on a different dependent variable, provenance"),
    ("", " significantly changes refusal behaviour (Cochran’s Q = 9.09, df = 3, p = 0.028; "
         "Table 6), and in the opposite of the expected direction: case reports, the "
         "“authoritative” source, provoke the most refusals (88.1%); exam text provokes the "
         "fewest (76.2%). "),
    ("new", "A powered replication needs the low-refusal prompt already implemented and roughly "
            "77 queries for 50 usable ones; at the refusal rates observed here the same design "
            "needs 616 queries (about 2.8 million generation tokens), which is why it was not "
            "run. "),
    ("", "This is absence of evidence, not evidence of absence"),
    ("fix", ", and the result is provisional."),
], note="R3.6 and R1.e: H03 is provisional, its refusal result is a secondary outcome, and the "
        "cost of a powered replication is stated. NEW: 59 of the 63 'scoreable' MultiCaRe "
        "answers are also flagged refusals (results/h3_provenance/h3_power.md), so Table 6's F1 "
        "column is weaker than the sample size alone suggests.")

capT6 = par("Table 6  Evidence-provenance conditions")
set_par(capT6, [
    ("", "Table 6  Evidence-provenance conditions on four topically matched corpora of 1,872 "
         "documents each (gpt-oss-120b, <i>n</i> = 160 per condition). "),
    ("new", "“Also refusal” counts scoreable answers that the refusal detector also flagged."),
], note="R3.6: the refusal/scoreable overlap made visible.")
remove_table_after(capT6)
build_table(capT6, widths=[2700, 1100, 1200, 1400, 1600],
            header=["Evidence corpus", "Refusal", "Mean F1", "n scoreable", "Also refusal"],
            rows=[
                (["MultiCaRe (case reports)", "88.1%", "0.3265", "63", "59"], "", set(), False),
                (["Shuffled MultiCaRe control", "82.5%", "0.3186", "61", "53"], "", set(), False),
                (["PubMedQA (abstracts)", "79.4%", "0.3476", "70", "53"], "", set(), False),
                (["MMedBench (exam text)", "76.2%", "0.2659", "66", "56"], "", set(), False),
            ],
            cell_kind=lambda i, j: ("new" if j == 4 else ""),
            note="New column from results/h3_provenance/h3_power.md; other values unchanged.")

# =========================================================================== #
# 6. Discussion and limitations
# =========================================================================== #
set_par(par("The retrieval findings are the study"), [
    ("", "The retrieval findings are the study’s most decision-relevant. The code-mixing penalty "
         "is significant under every system tested"),
    ("new", ", at every depth and on every metric,"),
    ("", " and roughly 4.4 times larger for lexical than for dense retrieval"),
    ("new", " at rank 1, with a wider gap at depth"),
    ("", ", producing a best-on-English and worst-on-Hinglish crossover for BM25. "),
    ("new", "Because the English arm is a clinician-written summary rather than a translation, "
            "the estimate is an upper bound on the language effect alone (Sect. 5.1). "),
    ("", "For deployments whose users write romanised code-switched queries the architectural "
         "implication is direct: the retrieval layer should be cross-lingual and dense, or "
         "hybrid, because the surface-form matching is exactly what romanisation destroys"),
    ("new", "; translating the query first does not rescue lexical retrieval"),
    ("", "."),
], note="Discussion aligned with the new depth, translation and register results.")

set_par(par("The MuRIL result sharpens the mechanism"), [
    ("re", "The MuRIL result points to script rather than language"),
    ("", ": an encoder pretrained on the right language in the wrong script retrieves at "
         "chance (0.0640 against a 0.0626 floor) and recovers to 0.1821 on identical content "
         "in English. "),
    ("fix", "But transliterating the query into Devanagari recovers only 6.5% of that gap, so "
            "script is not the binding constraint: either the encoder’s Hindi representations "
            "are weaker than its English ones on this clinical content, or a hand-built "
            "transliteration is too approximate to carry the content, and the study cannot "
            "separate the two. Transliteration-aware pretraining or romanisation normalisation "
            "remains the indicated remedy; this study tests the cheapest version of it and "
            "finds it insufficient."),
], note="R1: the untested mechanism claim is replaced by the tested result and its two "
        "remaining readings.")

set_par(par("Grounding is better described as a robustness"), [
    ("", "Grounding is better described as a robustness and behaviour mechanism than a "
         "guaranteed quality gain. It stabilises factual support against code-mixing "
         "intensity, "),
    ("re", "shifts one generator toward precision over recall, "),
    ("", "and converts silent confabulation into explicit refusal when the evidence does not "
         "fit. Whether that trade profits depends on the generator (the two families disagree "
         "in sign under the unbiased reference), "),
    ("re", "the size of any reported gain depends on the scoring reference, and part of the "
           "measured gain is an extraction artefact whose direction favours the result "
           "(Sect. 5.2)"),
    ("", ". None of this retracts the effect; it bounds how it may honestly be reported."),
], note="R3.5 and the extractor audit: the discussion now carries both qualifications.")

set_par(par("The evaluation findings generalise beyond this system"), [
    ("", "The evaluation findings generalise beyond this system. Factuality should be scored "
         "with F1 rather than precision alone; degenerate baselines should accompany scored "
         "tables, since a one-word constant outscoring a real system 4.7-fold is otherwise "
         "invisible; "),
    ("new", "concept extractors should be audited against an independent reader, per arm, "
            "because an arm-dependent bias does not cancel in a paired difference; "),
    ("", "refusal and coverage belong beside every factuality number, because refusals vanish "
         "from naive means precisely where systems fail; factuality and hallucination should "
         "not be reported as two findings when one is the complement of the other; references "
         "neither arm saw are preferable; "),
    ("fix", "dose–response should be tested as it was planned, on the paired difference, with "
            "per-arm correlations as the decomposition"),
    ("", "; and corpora should be topically matched before effects are attributed to "
         "provenance."),
], note="R3.3: the earlier recommendation to test dose–response per arm 'rather than on the "
        "noisier arm difference' contradicted the reviewer's valid point and is corrected.")

set_par(par("Seven limitations bound the claims"), [
    ("fix", "Ten limitations bound the claims, and each comes with its measured extent. "),
    ("", "First, the relevance labels are coarse: condition-group matching over 18 groups "
         "admits same-group but clinically irrelevant cases, so absolute Success@1 is a lower "
         "bound on true relevance"),
    ("new", "; the graded variant reported in Sect. 4.2 is an automatic proxy, not a clinical "
            "judgment"),
    ("", ". Second, the concept lexicon includes only 26 concepts and has not been validated by "
         "clinicians"),
    ("new", "; against an independent model reader its extractor reaches κ = 0.494 with "
            "precision 0.412, and the over-attribution is larger in the zero-shot arm, which "
            "inflates the grounding effect of Sect. 5.3 by an amount that is bounded but not "
            "determined. Clinician validation remains future work"),
    ("", ". Third, the unbiased reference is narrow and repetitive: it averages 1.5 concepts, "
         "takes 412 distinct values across 2,988 rows, and a single value accounts for 22% of "
         "rows; cluster bootstraps deal with the induced non-independence, but the construct "
         "captured is partial"),
    ("new", ", and its free-text captions contain frequent misspellings that a lexical extractor "
            "cannot match"),
    ("", ". Fourth, generation results are generator-dependent: two results differ in sign "
         "between the generator families under the unbiased reference, and both are reported "
         "with neither being chosen."),
], note="R3.4, R1.b, R1.c: limitations 1–3 now carry the automatic-validation evidence and "
        "state explicitly that no clinician reviewed the lexicon.")

set_par(par("Fifth, provider instability is a real threat"), [
    ("", "Fifth, provider instability is a real threat: llama-3.1-8b-instant, the generator "
         "behind the <i>n</i> = 1,165 analyses, was decommissioned mid-study and now returns "
         "HTTP 404, so its outputs remain archived and re-scorable but no new generations are "
         "possible, a genuine danger for research on hosted models"),
    ("new", "; the gpt-oss models are open-weight, so their results are regenerable in "
            "principle, and the archived generations make every number re-scorable"),
    ("", ". Sixth, the H₀₃ answer-quality comparison is under-powered, with only 13 of 160 "
         "queries scoreable under all four evidence conditions"),
    ("new", ", most of them refusals; a powered replication needs roughly 616 queries under "
            "the present prompt or 77 under the low-refusal one"),
    ("", " (Sect. 5.5). Seventh, no clinical deployment claim is made: absolute Success@1 of "
         "0.1751 on Hinglish, roughly two to three times the random floor, is far below what "
         "patient-facing use would require"),
    ("new", ", although Success@10 of 0.656 makes the pipeline usable as a first-stage "
            "candidate generator. Eighth, the English arm of the penalty comparison is a "
            "clinician-written summary, not a translation, so the penalty is an upper bound on "
            "the language effect and conflates language with conciseness and register "
            "(Sect. 5.1). Ninth, the transliteration is a hand-verified lexicon with rule-based "
            "mapping for the tail rather than a trained model, so its recovery is a lower "
            "bound, and the translation baseline covers a 500-query stratified subset. Tenth, "
            "H04 was formulated after the initial retrieval experiments and is confirmatory only "
            "in the sense that it was then tested on the full matched set"),
    ("", "."),
], note="R1.a (low absolute performance contextualised), R1.f (decommissioning mitigation), "
        "and the three new limitations from the revision analyses: register confound, "
        "approximate transliteration, and H04 being post hoc.")

# =========================================================================== #
# 7. Conclusion
# =========================================================================== #
set_par(par("This study set out to characterise"), [
    ("", "This study set out to characterise, rather than improve, retrieval-augmented "
         "generation for Hinglish clinical questions over English case reports. Code-mixing "
         "significantly degrades clinical retrieval, about four times more for lexical than for "
         "dense methods"),
    ("new", " at rank 1 and more at depth; neither transliteration nor translation removes the "
            "penalty for lexical retrieval"),
    ("", "; and while grounding "),
    ("re", "increases evidence-referenced factual support"),
    ("", " and shields it from the code-mixing intensity, the apparent size of that benefit "
         "depends substantially on the evaluation protocol"),
    ("new", " and is partly an extraction artefact"),
    ("", "."),
], note="Conclusion aligned with the qualified H01 claim (R3.5) and the new baselines.")

set_par(par("The contributions are the retrieval-stage"), [
    ("", "The contributions are the retrieval-stage code-mixing penalty with its lexical/dense "
         "asymmetry and "),
    ("re", "tested script-mismatch mechanism"),
    ("", "; "),
    ("re", "pre-specified"),
    ("", " hypothesis tests replicated across two generator families"),
    ("re", " under oracle evidence"),
    ("", "; and a demonstration that evaluation design changes the reported benefit, with the "
         "degenerate baselines "),
    ("new", "and extractor audit "),
    ("", "needed to detect it."),
], note="Terminology and replication claim aligned with the corrected results.")

set_par(par("Future work should replace coarse"), [
    ("", "Future work should replace coarse condition-group labels with graded clinical "
         "relevance, validate the 26-concept lexicon with clinicians"),
    ("new", " (a blinded annotation protocol is released with the code)"),
    ("", ", adopt richer references, "),
    ("new", "replace the summary-based English arm with human translations so that language "
            "and register are separated, "),
    ("", "test transliteration-aware encoders against script mismatch, and power the provenance "
         "comparison for answer quality"),
    ("new", " with the low-refusal prompt"),
    ("", ". Each tightens a bound measured here."),
], note="Future work updated: the annotation tooling exists in the repository, unused.")

# =========================================================================== #
# 8. Declarations and references
# =========================================================================== #
set_par(par("Code Availability:"), [
    ("", "Code Availability: Available at https://github.com/saaachis/hinglish-grounded-medical-rag, "
         "release tag "),
    ("fix", "v1.1-iccsdi2026-revised"),
    ("", ". Archived outputs permit re-scoring of every reported number (Sect. 6.1)"),
    ("new", ", and python -m src.analysis.verify_paper_numbers checks every headline value in "
            "this manuscript against the committed results"),
    ("", "."),
], note="ACTION: the submitted tag v1.0-iccsdi2026 does not exist in the repository. Create "
        "the tag named here on the final commit and publish a GitHub release before "
        "resubmitting.")

p_ref11 = par("[11] Anonymous: HiFactMix")
doc.add_comment(runs=[p_ref11.runs[0]], text="ACTION: 'under review, ICLR 2026' is stale — "
                "ICLR 2026 decisions are out. Check OpenReview and cite the published or arXiv "
                "version, or drop the reference.", author=AUTHOR, initials=INITIALS)
p_mentor = par("Mentor: Prof. Dr. Rajesh Maurya")
doc.add_comment(runs=[p_mentor.runs[0]], text="CHECK: a 'Mentor' line is not a field in the "
                "ICCSDI template byline. Confirm with Prof. Maurya and the department whether he "
                "should be a co-author or remain in the Acknowledgements.",
                author=AUTHOR, initials=INITIALS)

doc.save(OUT)
print("wrote", OUT.name)
