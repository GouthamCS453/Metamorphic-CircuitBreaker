import os
import re
import subprocess
import fitz
import markdown

ROOT_DIR = r"C:\Users\Goutham C S\Desktop\main_implementation"
MD_PATH = os.path.join(ROOT_DIR, "MCB_Portable_Results_Analysis_and_Methodology_Guide.md")
HTML_PATH = os.path.join(ROOT_DIR, "MCB_Portable_Results_Analysis_and_Methodology_Guide.html")
PDF_PATH = os.path.join(ROOT_DIR, "MCB_Portable_Results_Analysis_and_Methodology_Guide.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    raw_md = f.read()

# Protect display math ($$...$$) and inline math ($...$)
math_blocks = {}
counter = 0

def save_display_math(match):
    global counter
    token = f"KATEXDISPLAYTOKEN{counter}ENDTOKEN"
    math_blocks[token] = match.group(0)
    counter += 1
    return token

def save_inline_math(match):
    global counter
    token = f"KATEXINLINETOKEN{counter}ENDTOKEN"
    math_blocks[token] = match.group(0)
    counter += 1
    return token

# Replace display math first
text_protected = re.sub(r"\$\$([\s\S]+?)\$\$", save_display_math, raw_md)
# Replace inline math (exclude escaped dollars and ensure non-empty)
text_protected = re.sub(r"(?<!\\)\$(?!\$)([^\$\n]+?)(?<!\\)\$", save_inline_math, text_protected)

# Convert Markdown to HTML
html_body = markdown.markdown(
    text_protected,
    extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"]
)

# Restore protected math tokens
for token, raw_math in math_blocks.items():
    html_body = html_body.replace(token, raw_math)

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
        content: counter(page);
        font-size: 8pt;
        color: #64748b;
    }
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.55;
    font-size: 9pt;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 16pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
    letter-spacing: -0.3px;
    line-height: 1.25;
}

h2 {
    font-size: 12pt;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h3 {
    font-size: 10pt;
    font-weight: 700;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

h4 {
    font-size: 9.5pt;
    font-weight: 600;
    color: #475569;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    margin-bottom: 14px;
    font-size: 8pt;
    page-break-inside: avoid;
}

th, td {
    padding: 5px 7px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}

th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    border-top: 1px solid #cbd5e1;
    border-bottom: 1.5px solid #cbd5e1;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

code {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 8pt;
    background-color: #f1f5f9;
    padding: 1.5px 3.5px;
    border-radius: 3px;
    color: #0f172a;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #2563eb;
    padding: 8px 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 7.5pt;
    line-height: 1.35;
    margin-top: 6px;
    margin-bottom: 10px;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #1e293b;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 16px 0;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
}

li {
    margin-bottom: 3px;
}

/* KaTeX custom presentation styling */
.katex-display {
    margin: 10px 0 !important;
    padding: 8px 12px !important;
    background-color: #f8fafc;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    overflow-x: auto;
}

.katex {
    font-size: 1.02em !important;
}
</style>
"""

katex_headers = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    });
</script>
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Metamorphic Circuit Breaker (MCB) Portable Test Suite: Comprehensive Analysis Guide</title>
    {custom_css}
    {katex_headers}
</head>
<body>
    {html_body}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)

print("HTML saved to:", HTML_PATH)

browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
abs_html = "file:///" + HTML_PATH.replace("\\", "/")
abs_pdf = os.path.abspath(PDF_PATH)

cmd = [
    browser_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=8000",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={abs_pdf}",
    abs_html
]

print("Rendering math and printing PDF...")
res = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 0:
    print(f"SUCCESS: PDF generated at: {abs_pdf}")
    print(f"Size: {os.path.getsize(abs_pdf):,} bytes")
    doc = fitz.open(abs_pdf)
    print(f"Pages: {len(doc)}")
    # Sample check Section 1.1 on page 1/2
    for page_idx in range(min(3, len(doc))):
        page_text = doc[page_idx].get_text()
        print(f"--- PAGE {page_idx + 1} SAMPLE ---")
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for line in lines[:8]:
            try:
                print(" ", line.encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass
else:
    print("FAILED to generate PDF.")
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
