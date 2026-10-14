import os
import re
import subprocess
import markdown

ROOT_DIR = r"C:\Users\Goutham C S\Desktop\main_implementation"
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "evaluation_report")
MD_PATH = os.path.join(OUTPUT_DIR, "Comprehensive_Cross_Domain_Evaluation_Report.md")
HTML_PATH = os.path.join(OUTPUT_DIR, "Comprehensive_Cross_Domain_Evaluation_Report.html")
PDF_PATH = os.path.join(OUTPUT_DIR, "Metamorphic_Circuit_Breaker_Cross_Domain_Evaluation_Report.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_content = f.read()

# Replace relative image paths like 'figures/fig1_domain_radar.png' with absolute file URIs
fig_dir_abs = os.path.join(OUTPUT_DIR, "figures").replace("\\", "/")
md_content = md_content.replace('src="figures/', f'src="file:///{fig_dir_abs}/')

# Protect display and inline math
math_store = {}
def protect_display_math(match):
    key = f"XYZDISPLAYMATH{len(math_store)}XYZ"
    math_store[key] = match.group(0)
    return key

def protect_inline_math(match):
    key = f"XYZINLINEMATH{len(math_store)}XYZ"
    math_store[key] = match.group(0)
    return key

content_protected = re.sub(r"\$\$([\s\S]+?)\$\$", protect_display_math, md_content)
content_protected = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", protect_inline_math, content_protected)

html_body = markdown.markdown(
    content_protected,
    extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"]
)

for key, raw_math in math_store.items():
    html_body = html_body.replace(key, raw_math)

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

@page {
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-right {
        content: counter(page);
        font-size: 8pt;
        color: #94a3b8;
    }
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.55;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 17pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #4f46e5;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 10px;
    letter-spacing: -0.3px;
}

h2 {
    font-size: 12.5pt;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #4f46e5;
    padding-left: 8px;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h3 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 14px 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 10px 12px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 7.5pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 10px 0;
}

code {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 8pt;
    background-color: #f1f5f9;
    color: #4338ca;
    padding: 2px 4px;
    border-radius: 3px;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 18px;
}

li {
    margin-bottom: 3px;
}

div {
    page-break-inside: avoid;
}

img {
    page-break-inside: avoid;
}

.katex-display {
    margin: 10px 0 !important;
    padding: 8px 12px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 3px solid #6366f1 !important;
    border-radius: 5px !important;
    overflow-x: auto;
    page-break-inside: avoid;
}

.katex {
    font-size: 1.05em !important;
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
            throwOnError : false
        });
    });
</script>
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Metamorphic Circuit Breaker Comprehensive Cross-Domain Evaluation Report</title>
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

print("HTML generated at:", HTML_PATH)

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
browser_exe = edge_path if os.path.exists(edge_path) else chrome_path

abs_html = "file:///" + HTML_PATH.replace("\\", "/")

cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=8000",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    abs_html
]

print(f"Rendering math and printing to PDF via {os.path.basename(browser_exe)}...")
res = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
    print(f"SUCCESS: PDF generated at: {PDF_PATH}")
    print(f"File size: {os.path.getsize(PDF_PATH)} bytes")
else:
    print("FAILED to generate PDF.")
    print("Stdout:", res.stdout)
    print("Stderr:", res.stderr)
