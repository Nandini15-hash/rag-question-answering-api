"""One-off script: render EXPLANATION.md content as a single-page PDF.

Not part of the running app. Content is hand-transcribed from EXPLANATION.md
(kept in sync manually) rather than parsed, since the layout needs a table
for the failure-case comparison and tight, specific spacing to fit one page.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT = Path(__file__).parent.parent / "EXPLANATION.pdf"

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=14, leading=16, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Oblique", fontSize=8.5, leading=10, spaceAfter=8, textColor=colors.HexColor("#444444")),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1a1a1a")),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.7, leading=11, spaceAfter=3, alignment=4),  # justify
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=8.7, leading=11, spaceAfter=2, leftIndent=12, bulletIndent=2),
    "tablehead": ParagraphStyle("tablehead", fontName="Helvetica-Bold", fontSize=7.8, leading=9.5, textColor=colors.white),
    "tablecell": ParagraphStyle("tablecell", fontName="Helvetica", fontSize=7.8, leading=9.5),
}

B = lambda s: f"<b>{s}</b>"

story = []
story.append(Paragraph("Explanation", styles["title"]))
story.append(Paragraph("RAG Question-Answering API &mdash; one page, covering the four required points.", styles["subtitle"]))

# --- 1 ---
story.append(Paragraph("1. Design parameter: chunk size (800 chars, 120 overlap)", styles["h2"]))
story.append(Paragraph(
    "The sample corpus (HR-handbook-style text) is short, self-contained paragraphs, ~100&ndash;500 characters "
    "each. 800 characters keeps one chunk to roughly one topic (e.g. the whole Leave Policy paragraph, including "
    "its three related facts: accrual rate, annual cap, and carry-over rule), so a single retrieved chunk "
    "usually contains a complete answer instead of a fragment. Smaller chunks (~200 chars) would split that "
    "paragraph across two or three vectors, none of which alone answers the question, forcing retrieval to rank "
    "multiple fragments correctly &mdash; a harder problem. Larger chunks (2000+) start mixing unrelated topics "
    "into one embedding, diluting it. 120-character overlap (15%) exists so a fact sitting near a chunk boundary "
    "is still retrievable from either side of it. <font name='Helvetica-Oblique'>chunk_size</font> and "
    "<font name='Helvetica-Oblique'>chunk_overlap</font> are both .env-configurable "
    "(app/config.py); a corpus of longer narrative text would want a larger value.",
    styles["body"],
))

# --- 2 ---
story.append(Paragraph("2. Failure actually observed: threshold can't separate relevant from irrelevant queries", styles["h2"]))
story.append(Paragraph(
    "The task requires saying “not in the documents” instead of inventing an answer. My first "
    "implementation used one cosine-similarity cutoff on the top retrieved chunk. Testing it with the "
    "zero-network <font name='Helvetica-Oblique'>offline</font> embedder (hashing-based bag-of-words, used "
    "because this sandbox's egress policy blocks huggingface.co, the source of the recommended "
    "<font name='Helvetica-Oblique'>local</font> sentence-transformers model &mdash; confirmed via a 403 policy "
    "denial, not a bug) surfaced this:",
    styles["body"],
))

table_data = [
    [Paragraph("Query (relevant to the docs)", styles["tablehead"]), Paragraph("top sim.", styles["tablehead"]),
     Paragraph("Query (irrelevant)", styles["tablehead"]), Paragraph("top sim.", styles["tablehead"])],
    [Paragraph("“How many paid leave days...?”", styles["tablecell"]), Paragraph("0.171", styles["tablecell"]),
     Paragraph("“Capital of France?”", styles["tablecell"]), Paragraph("0.213", styles["tablecell"])],
    [Paragraph("“Deadline for expense claims?”", styles["tablecell"]), Paragraph("0.275", styles["tablecell"]),
     Paragraph("“How to bake a cake?”", styles["tablecell"]), Paragraph("0.133", styles["tablecell"])],
    [Paragraph("“On-call compensation?”", styles["tablecell"]), Paragraph("0.238", styles["tablecell"]),
     Paragraph("“2011 cricket World Cup winner?”", styles["tablecell"]), Paragraph("0.201", styles["tablecell"])],
]
tbl = Table(table_data, colWidths=[2.15 * inch, 0.55 * inch, 2.15 * inch, 0.55 * inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a4a4a")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
]))
story.append(Spacer(1, 2))
story.append(tbl)
story.append(Spacer(1, 3))

story.append(Paragraph(
    f"The ranges overlap &mdash; an irrelevant query (0.213) can score <i>higher</i> than a genuinely relevant "
    f"one (0.171). {B('Cause:')} the hashing embedder is purely lexical (token/token-pair overlap in a "
    f"fixed-size hashed space); it has no notion that “deadline” and “within 30 days” are "
    f"related, so incidental word overlap with an unrelated document can outscore real semantic relevance. "
    f"{B('Fix applied:')} a hard low floor (0.05) catches true non-matches cheaply for every mode; above that, "
    f"the OpenAI generation path defers to the LLM's own judgment of the retrieved text (reliable &mdash; it can "
    f"read), while only the no-LLM extractive fallback still depends on the raw threshold (documented as "
    f"unreliable specifically for the offline provider; the local/openai embedders' better-separated score "
    f"distributions weren't testable in this sandbox for the reason above).",
    styles["body"],
))

# --- 3 ---
story.append(Paragraph("3. Metric tracked: retrieval latency (+ top similarity score, logged alongside)", styles["h2"]))
story.append(Paragraph(
    "Every query logs retrieval_latency_ms, generation_latency_ms, and top_similarity_score to "
    "data/metrics.jsonl (app/utils/metrics.py). Across real test queries against a 2-document corpus, "
    f"retrieval consistently took {B('under 1ms')} &mdash; FAISS IndexFlatIP brute-force search over a handful "
    "of vectors is essentially free; almost all end-to-end latency (once an LLM key is set) is the generation "
    "call, not retrieval. That tells me this system's latency won't meaningfully change with corpus growth "
    "until the vector count is large enough (tens of thousands+) for exact search itself to become the "
    "bottleneck &mdash; at which point top_similarity_score, tracked alongside, is exactly the signal needed to "
    "check whether switching to an approximate index (IVF/HNSW) is trading away too much recall for speed.",
    styles["body"],
))

# --- 4 ---
story.append(Paragraph("4. Not finished, and what's next", styles["h2"]))
bullets = [
    f"{B('No numerical retrieval eval.')} I verified retrieval qualitatively (see &sect;2) but never built a fixed question&rarr;expected-chunk labeled set to compute precision/recall. Next: 15&ndash;20 labeled Q&amp;A pairs per document, scored automatically on every change to chunking/embedding config.",
    f"{B('local embedder untested here.')} The recommended default (sentence-transformers) couldn't run in this network-restricted sandbox; all testing used the offline fallback. Next: run the same eval set against local and openai on an unrestricted machine and tune SIMILARITY_THRESHOLD per provider from real numbers instead of a guess.",
    f"{B('No document replace/dedup.')} Re-uploading a same-named file creates a new document_id; old chunks aren't removed from the index. Next: a DELETE /documents/{{id}} endpoint that removes its vectors and metadata.",
    f"{B('No streaming.')} /query blocks until the full LLM response is ready. Next: stream tokens via SSE for better perceived latency on longer answers.",
]
for b in bullets:
    story.append(Paragraph(f"&bull;&nbsp; {b}", styles["bullet"]))

doc = SimpleDocTemplate(
    str(OUT), pagesize=letter,
    topMargin=0.45 * inch, bottomMargin=0.45 * inch,
    leftMargin=0.55 * inch, rightMargin=0.55 * inch,
    title="Explanation - RAG Question-Answering API",
)
doc.build(story)
print(f"Wrote {OUT}")
