"""One-off script: generate a sample PDF for demo/testing (not part of the app)."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

TEXT = """Sydon Forge Product Overview

Sydon Forge is an automated coding agent platform focused on governance and orchestration around AI coding tools. It sits between engineering teams and the growing ecosystem of AI coding assistants, giving organizations one control plane for policy, auditability, and cost.

Architecture Overview
Forge is composed of three layers. The orchestration layer routes tasks to the appropriate AI coding agent based on task type, repository, and configured policy. The governance layer enforces guardrails such as which repositories an agent may write to, mandatory human review for production branches, and secret-scanning before any commit is proposed. The observability layer records every agent action, prompt, and generated diff for audit purposes, retained for a configurable period (default 90 days).

Supported Agents
Forge integrates with multiple underlying coding agents through a plugin interface. Each plugin declares its capabilities (read-only search, code generation, test execution) and Forge enforces a consistent permission model across all of them, regardless of which underlying agent is running.

Deployment Model
Forge can be deployed as a managed cloud service or self-hosted within a customer's VPC. Self-hosted deployments require a PostgreSQL database for job state and an object store (S3-compatible) for audit logs. The managed cloud offering handles both automatically and includes a 99.9% uptime SLA.

Pricing
Forge is priced per active seat per month, with volume discounts starting at 50 seats. A seat is defined as any engineer who triggers at least one agent task in a given billing month. There is no charge for read-only observability access.

Security and Compliance
Forge underwent a SOC 2 Type II audit in the most recent fiscal year. All agent-generated code changes pass through the same CI/CD gates as human-authored changes, with no bypass mechanism. Customer source code is never used to train any underlying model; Forge operates as a pass-through orchestrator only.

Roadmap
The next major release focuses on multi-agent collaboration, allowing two or more agents to work on interdependent parts of the same task with automatic conflict resolution at the governance layer. A closed beta is planned to begin in the following quarter, with general availability targeted roughly two quarters after that.
"""

def make_pdf(path: Path):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    x_margin = 0.9 * inch
    y = height - 0.9 * inch
    c.setFont("Helvetica", 10)
    for paragraph in TEXT.strip().split("\n\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if c.stringWidth(test_line, "Helvetica", 10) > (width - 2 * x_margin):
                c.drawString(x_margin, y, line)
                y -= 14
                line = word
                if y < 0.9 * inch:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 0.9 * inch
            else:
                line = test_line
        if line:
            c.drawString(x_margin, y, line)
            y -= 14
        y -= 10  # paragraph gap
        if y < 0.9 * inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 0.9 * inch
    c.save()

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "sample_docs" / "forge_product_overview.pdf"
    make_pdf(out)
    print(f"Wrote {out}")
