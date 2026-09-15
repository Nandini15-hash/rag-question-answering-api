"""One-off script: screenshot diagrams/architecture.svg to PNG via headless Chromium."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
SVG = ROOT / "diagrams" / "architecture.svg"
PNG = ROOT / "diagrams" / "architecture.png"
HTML = ROOT / "diagrams" / "_tmp_render.html"

svg_content = SVG.read_text()
HTML.write_text(f"<html><body style='margin:0'>{svg_content}</body></html>")

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    page = browser.new_page(viewport={"width": 1030, "height": 670})
    page.goto(f"file://{HTML}")
    page.screenshot(path=str(PNG))
    browser.close()

HTML.unlink()
print(f"Wrote {PNG}")
