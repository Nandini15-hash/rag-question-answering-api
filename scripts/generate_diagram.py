"""
Generates diagrams/architecture.drawio (mxGraph XML, importable at
app.diagrams.net) programmatically, so the layout is defined once and stays
consistent / easy to regenerate. Not part of the running application.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).parent.parent / "diagrams" / "architecture.drawio"

NODE_STYLE = "rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=12;"
GROUP_STYLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#666666;verticalAlign=top;fontSize=13;fontStyle=1;"
EDGE_STYLE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;fontSize=11;{extra}"

COLORS = {
    "client": ("#dae8fc", "#6c8ebf"),
    "api": ("#d5e8d4", "#82b366"),
    "ingest": ("#ffe6cc", "#d79b00"),
    "store": ("#f8cecc", "#b85450"),
    "query": ("#e1d5e7", "#9673a6"),
}

nodes = []  # (id, x, y, w, h, label, colorkey)
edges = []  # (id, source, target, label, dashed)
_id = 0


def nid():
    global _id
    _id += 1
    return f"n{_id}"


def add_node(x, y, w, h, label, colorkey):
    i = nid()
    nodes.append((i, x, y, w, h, label, colorkey))
    return i


def add_edge(src, tgt, label="", dashed=False):
    i = nid()
    edges.append((i, src, tgt, label, dashed))
    return i


# --- Client ---
client = add_node(420, 20, 200, 50, "Client\n(curl / Swagger UI / frontend)", "client")

# --- API layer ---
upload_ep = add_node(120, 130, 220, 50, "POST /documents/upload\n(Pydantic validated, rate-limited)", "api")
status_ep = add_node(120, 200, 220, 50, "GET /documents/{job_id}/status", "api")
list_ep = add_node(120, 270, 220, 50, "GET /documents", "api")
query_ep = add_node(700, 130, 220, 50, "POST /query\n(Pydantic validated, rate-limited)", "api")

# --- Ingestion pipeline ---
save_file = add_node(400, 130, 220, 50, "Save upload to\ndata/uploads/", "ingest")
job_store = add_node(400, 200, 220, 50, "Create job row\njobs.db (SQLite)", "store")
worker = add_node(400, 270, 220, 50, "ThreadPoolExecutor worker\npicks up queued job", "ingest")
loader = add_node(400, 340, 220, 50, "Loader: pypdf (.pdf)\nor plain read (.txt)", "ingest")
chunker = add_node(400, 410, 220, 50, "Recursive chunker\n(chunk_size=800, overlap=120)", "ingest")
embedder_ingest = add_node(400, 480, 220, 50, "Embedder\n(local / openai / offline)", "ingest")

# --- Vector store ---
faiss_store = add_node(400, 550, 220, 60, "FAISS IndexFlatIP\n+ metadata.json\n(data/index/, persisted to disk)", "store")

# --- Query pipeline ---
embed_query = add_node(700, 200, 220, 50, "Embed question\n(same embedder as ingestion)", "query")
similarity = add_node(700, 270, 220, 50, "FAISS similarity search\n(top_k nearest chunks)", "query")
generation = add_node(700, 340, 220, 50, "Answer generation:\nOpenAI chat completion,\nor extractive fallback", "query")
metrics = add_node(700, 410, 220, 50, "Log metrics.jsonl\n(latency, top similarity, mode)", "store")
response = add_node(700, 480, 220, 50, "Response: answer +\nsources + metrics (JSON)", "query")

# --- Edges: client to API ---
add_edge(client, upload_ep)
add_edge(client, query_ep)
add_edge(client, status_ep, dashed=True)
add_edge(client, list_ep, dashed=True)

# --- Ingestion flow ---
add_edge(upload_ep, save_file)
add_edge(save_file, job_store, "202 Accepted\n(job_id returned)")
add_edge(job_store, worker, "picked up by\nworker pool")
add_edge(worker, loader)
add_edge(loader, chunker, "raw text")
add_edge(chunker, embedder_ingest, "chunks")
add_edge(embedder_ingest, faiss_store, "vectors +\nmetadata")
add_edge(status_ep, job_store, "reads status", dashed=True)
add_edge(list_ep, job_store, "reads latest\njob per doc", dashed=True)
add_edge(worker, job_store, "updates status:\nqueued -> processing -> done/failed", dashed=True)

# --- Query flow ---
add_edge(query_ep, embed_query)
add_edge(embed_query, similarity)
add_edge(similarity, faiss_store, "reads index", dashed=True)
add_edge(similarity, generation, "top-k chunks")
add_edge(generation, metrics)
add_edge(metrics, response)
add_edge(response, query_ep, "200 OK", dashed=True)


def build_xml():
    cells = []
    cells.append('<mxCell id="0" />')
    cells.append('<mxCell id="1" parent="0" />')

    for i, x, y, w, h, label, colorkey in nodes:
        fill, stroke = COLORS[colorkey]
        style = NODE_STYLE.format(fill=fill, stroke=stroke)
        cells.append(
            f'<mxCell id="{i}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
        )

    for i, src, tgt, label, dashed in edges:
        extra = "dashed=1;" if dashed else ""
        style = EDGE_STYLE.format(extra=extra)
        cells.append(
            f'<mxCell id="{i}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry" /></mxCell>'
        )

    body = "\n".join(cells)
    return f'''<mxfile host="app.diagrams.net" agent="rag-qa-system/generate_diagram.py" version="24.7.0">
  <diagram name="RAG Architecture" id="rag-architecture">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="700" math="0" shadow="0">
      <root>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''


SVG_OUT = Path(__file__).parent.parent / "diagrams" / "architecture.svg"


def build_svg():
    colors_hex = COLORS
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="650" '
        'viewBox="0 0 960 650" font-family="Helvetica, Arial, sans-serif">',
        '<rect width="960" height="650" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#555555"/></marker></defs>',
    ]

    node_map = {i: (x, y, w, h) for i, x, y, w, h, label, colorkey in nodes}

    # edges first (so nodes draw on top)
    for i, src, tgt, label, dashed in edges:
        sx, sy, sw, sh = node_map[src]
        tx, ty, tw, th = node_map[tgt]
        x1, y1 = sx + sw / 2, sy + sh
        x2, y2 = tx + tw / 2, ty
        # simple elbow if far apart horizontally and roughly same row band
        if abs((sy + sh / 2) - (ty + th / 2)) < 5 and sx != tx:
            x1, y1 = (sx + sw, sy + sh / 2) if sx < tx else (sx, sy + sh / 2)
            x2, y2 = (tx, ty + th / 2) if sx < tx else (tx + tw, ty + th / 2)
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#555555" stroke-width="1.5" fill="none"{dash} marker-end="url(#arrow)"/>')
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            lines = label.split("\n")
            for li, line in enumerate(lines):
                parts.append(
                    f'<text x="{mx}" y="{my - 4 + li * 11}" font-size="9.5" fill="#333333" '
                    f'text-anchor="middle" style="paint-order:stroke;stroke:#ffffff;stroke-width:3px">{escape(line)}</text>'
                )

    for i, x, y, w, h, label, colorkey in nodes:
        fill, stroke = colors_hex[colorkey]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        lines = label.split("\n")
        n = len(lines)
        start_y = y + h / 2 - (n - 1) * 7 + 4
        for li, line in enumerate(lines):
            parts.append(
                f'<text x="{x + w / 2}" y="{start_y + li * 14}" font-size="11.5" fill="#222222" '
                f'text-anchor="middle" font-weight="{600 if li == 0 and n == 1 else 400}">{escape(line)}</text>'
            )

    # legend
    legend_items = [
        ("Client", "client"),
        ("API layer", "api"),
        ("Ingestion pipeline", "ingest"),
        ("Storage", "store"),
        ("Query pipeline", "query"),
    ]
    lx, ly = 20, 600
    for label, key in legend_items:
        fill, stroke = colors_hex[key]
        parts.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{lx + 22}" y="{ly + 13}" font-size="11" fill="#222222">{escape(label)}</text>')
        lx += 22 + len(label) * 6.2 + 20

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(build_xml())
    print(f"Wrote {OUT}")
    SVG_OUT.write_text(build_svg())
    print(f"Wrote {SVG_OUT}")
