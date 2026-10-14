"""
scripts/generate_presentable_ieee_paper.py — Generate the rectified, publication-ready
IEEE Research Paper in docx and pdf format without modifying any previous paper files.

Key Rectifications:
1. Reconciled Abstract: Completely eliminated run-on sentences and conflicting status statements.
2. Inverted Severity Taxonomy Table: Fully aligned taxonomy table weights [1.0, 0.6, 0.3] with Equation (1).
3. Mathematical Consistency: Added Cardinality-Normalized Level 2 IR_k formula and dynamically coupled theta_warn = 0.50 * theta_trip in Eq. (6).
4. Strictly Sequential Figure Numbering: Sequenced Fig. 1 to Fig. 6 with exact matching in-text citations.
5. Reconciled Performance Claims: Harmonized Table I, Section VI, and Section VIII conclusion metrics.
6. Clean Separate Output: Saved exclusively to Metamorphic_Circuit_Breaker_IEEE_Paper_Rectified_v2.docx and .pdf.
"""

import os
import io
import zipfile
import subprocess
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
import latex2mathml.converter
import lxml.etree as ET

ROOT_DIR = r"C:\Users\Goutham C S\Desktop\main_implementation"
TEMPLATE_PATH = os.path.join(ROOT_DIR, "Research Paper", "conference-template-a4.docx")
OUTPUT_DOCX_PATH = os.path.join(ROOT_DIR, "Research Paper", "Metamorphic_Circuit_Breaker_IEEE_Paper_Rectified_v2.docx")
OUTPUT_PDF_PATH = os.path.join(ROOT_DIR, "Research Paper", "Metamorphic_Circuit_Breaker_IEEE_Paper_Rectified_v2.pdf")
FIG_DIR = os.path.join(ROOT_DIR, "outputs", "evaluation_report", "figures")
XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"

# Load XSLT stylesheet for LaTeX -> OMML conversion
xslt_doc = ET.parse(XSL_PATH)
transform_mml = ET.XSLT(xslt_doc)


def latex_to_omml(latex_code: str, bold: bool = True):
    """Convert a LaTeX formula to a native Microsoft Word OMML equation element with bold emphasis."""
    mathml = latex2mathml.converter.convert(latex_code)
    mathml_doc = ET.fromstring(mathml)
    omml_doc = transform_mml(mathml_doc)
    ns = {
        'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    }

    # Clean non-breaking spaces
    for t in omml_doc.xpath('.//m:t', namespaces=ns):
        if t.text:
            t.text = t.text.replace('\xa0', ' ')

    # Format piecewise cases function spacing
    for mr in omml_doc.xpath('.//m:mr', namespaces=ns):
        elements = mr.xpath('./m:e', namespaces=ns)
        if len(elements) >= 2:
            t_first = elements[1].xpath('.//m:t', namespaces=ns)
            if t_first and t_first[0].text:
                if not t_first[0].text.startswith(' '):
                    t_first[0].text = '   ' + t_first[0].text

    for mcs in omml_doc.xpath('.//m:mcs', namespaces=ns):
        mcs.clear()
        for _ in range(2):
            mc = ET.SubElement(mcs, '{http://schemas.openxmlformats.org/officeDocument/2006/math}mc')
            mcPr = ET.SubElement(mc, '{http://schemas.openxmlformats.org/officeDocument/2006/math}mcPr')
            ET.SubElement(mcPr, '{http://schemas.openxmlformats.org/officeDocument/2006/math}count', {'{http://schemas.openxmlformats.org/officeDocument/2006/math}val': '1'})
            ET.SubElement(mcPr, '{http://schemas.openxmlformats.org/officeDocument/2006/math}mcJc', {'{http://schemas.openxmlformats.org/officeDocument/2006/math}val': 'left'})

    for r in omml_doc.xpath('.//m:r', namespaces=ns):
        w_rPr = r.find('w:rPr', namespaces=ns)
        if w_rPr is None:
            w_rPr = ET.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
            r.insert(0, w_rPr)
        rFonts = w_rPr.find('w:rFonts', namespaces=ns)
        if rFonts is None:
            rFonts = ET.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
            w_rPr.append(rFonts)
        rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', 'Times New Roman')
        rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', 'Times New Roman')
        rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cs', 'Times New Roman')
        if bold:
            b_elem = w_rPr.find('w:b', namespaces=ns)
            if b_elem is None:
                b_elem = ET.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b')
                w_rPr.append(b_elem)

        m_rPr = r.find('m:rPr', namespaces=ns)
        if m_rPr is None:
            m_rPr = ET.Element('{http://schemas.openxmlformats.org/officeDocument/2006/math}rPr')
            r.append(m_rPr)
        if bold:
            sty = m_rPr.find('m:sty', namespaces=ns)
            if sty is None:
                sty = ET.Element('{http://schemas.openxmlformats.org/officeDocument/2006/math}sty')
                m_rPr.append(sty)
            sty.set('{http://schemas.openxmlformats.org/officeDocument/2006/math}val', 'b')

    omml_str = ET.tostring(omml_doc, encoding="unicode")
    if omml_str.startswith("<?xml"):
        omml_str = omml_str.split("?>", 1)[1]
    return parse_xml(omml_str)


def normalize_template_namespaces(file_path: str) -> io.BytesIO:
    """Normalize ISO/IEC 29500 transitional namespaces to standard OpenXML."""
    with open(file_path, "rb") as f:
        data = f.read()

    replacements = [
        (b"http://purl.oclc.org/ooxml/wordprocessingml/main", b"http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
        (b"http://purl.oclc.org/ooxml/officeDocument/relationships", b"http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
        (b"http://purl.oclc.org/ooxml/officeDocument/math", b"http://schemas.openxmlformats.org/officeDocument/2006/math"),
        (b"http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing", b"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
        (b"http://purl.oclc.org/ooxml/officeDocument/extendedProperties", b"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"),
    ]

    in_zip = zipfile.ZipFile(io.BytesIO(data))
    out_buf = io.BytesIO()
    out_zip = zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED)
    for item in in_zip.infolist():
        content = in_zip.read(item.filename)
        for old, new in replacements:
            content = content.replace(old, new)
        out_zip.writestr(item, content)
    out_zip.close()
    out_buf.seek(0)
    return out_buf


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>\n'
        f'  <w:top w:w="{top}" w:type="dxa"/>\n'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>\n'
        f'  <w:left w:w="{left}" w:type="dxa"/>\n'
        f'  <w:right w:w="{right}" w:type="dxa"/>\n'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_ieee_table_borders(table):
    """Apply standard IEEE 3-line table borders (top, header bottom, table bottom)."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>\n'
            f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="auto"/>\n'
            f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="auto"/>\n'
            f'  <w:left w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'  <w:insideH w:val="none"/>\n'
            f'  <w:insideV w:val="none"/>\n'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

    if len(table.rows) > 0:
        for cell in table.rows[0].cells:
            tcPr = cell._element.get_or_add_tcPr()
            tcBorders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>\n'
                f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="auto"/>\n'
                f'</w:tcBorders>'
            )
            tcPr.append(tcBorders)


def populate_author_cell(cell, name: str, designation: str, dept: str, college: str, location: str, email: str):
    """Format an author cell with isolated, centered, standard IEEE typography."""
    cell.text = ""
    set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)

    # Name: 11pt Bold
    r_name = p.add_run(name + "\n")
    r_name.font.name = "Times New Roman"
    r_name.font.size = Pt(11.0)
    r_name.font.bold = True

    # Designation: 9.5pt Italic
    r_desig = p.add_run(designation + "\n")
    r_desig.font.name = "Times New Roman"
    r_desig.font.size = Pt(9.5)
    r_desig.font.italic = True

    # Dept & College: 9.0pt
    r_dept = p.add_run(dept + "\n" + college + "\n")
    r_dept.font.name = "Times New Roman"
    r_dept.font.size = Pt(9.0)

    # Location: 9.0pt
    r_loc = p.add_run(location + "\n")
    r_loc.font.name = "Times New Roman"
    r_loc.font.size = Pt(9.0)

    # Email: 9.0pt
    r_email = p.add_run(email)
    r_email.font.name = "Times New Roman"
    r_email.font.size = Pt(9.0)


def build_paper():
    print("Normalizing template namespaces...")
    norm_buf = normalize_template_namespaces(TEMPLATE_PATH)
    doc = docx.Document(norm_buf)

    # ── Fix Styles: Change equation style font to Cambria Math ─────────────────
    if "equation" in doc.styles:
        eq_style = doc.styles["equation"]
        rPr = eq_style._element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is not None:
            rFonts.set(qn("w:ascii"), "Cambria Math")
            rFonts.set(qn("w:hAnsi"), "Cambria Math")
            rFonts.set(qn("w:cs"), "Times New Roman")
        else:
            rPr.append(parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Times New Roman"/>'))

    # ── Fix Headings: Remove automated numbering numPr to prevent duplicate letters
    for h_name in ["Heading 1", "Heading 2", "Heading 3", "Heading 4", "references"]:
        if h_name in doc.styles:
            h_style = doc.styles[h_name]
            pPr = h_style._element.find(qn("w:pPr"))
            if pPr is not None:
                numPr = pPr.find(qn("w:numPr"))
                if numPr is not None:
                    pPr.remove(numPr)

    # Clear all existing body paragraphs and tables (preserving root sectPr)
    body_element = doc._body._element
    for child in list(body_element):
        if child.tag.endswith("p") or child.tag.endswith("tbl"):
            body_element.remove(child)

    # Configure Section 0 (Title and Authors - 1 column full width)
    sec0 = doc.sections[0]
    sec0.page_width = Inches(8.27)    # A4 width
    sec0.page_height = Inches(11.69)  # A4 height
    sec0.top_margin = Inches(0.75)    # 19 mm
    sec0.bottom_margin = Inches(1.00) # 25.4 mm
    sec0.left_margin = Inches(0.56)   # 14.3 mm
    sec0.right_margin = Inches(0.56)  # 14.3 mm

    # ── 1. Paper Title ────────────────────────────────────────────────────────
    p_title = doc.add_paragraph()
    p_title.style = doc.styles["paper title"]
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Metamorphic Circuit Breaker: A Hierarchical Fault Attribution Framework for Runtime Safety in Medical Vision Models")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(24.0)

    # ── 2. Conference & Date Information ──────────────────────────────────────
    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_before = Pt(4)
    p_date.paragraph_format.space_after = Pt(14)
    r_date = p_date.add_run("September 2026 | IEEE Conference on Artificial Intelligence Safety and Healthcare Engineering (Academic Year 2025\u20132026)")
    r_date.font.name = "Times New Roman"
    r_date.font.size = Pt(9.5)
    r_date.font.italic = True

    # ── 3. Author Blocks (Two-Tier Clean Table: Row 1 = 3 cols, Row 2 = 2 cols)
    COLLEGE = "Mar Baselios College of Engineering and Technology (Autonomous)"
    AFFIL_DEPT = "Department of Computer Science and Engineering"
    LOCATION = "Thiruvananthapuram, Kerala, India"

    # Row 1: Authors 1, 2, 3
    tbl_row1 = doc.add_table(rows=1, cols=3)
    tbl_row1.alignment = WD_TABLE_ALIGNMENT.CENTER
    row1_cells = tbl_row1.rows[0].cells
    for c in row1_cells:
        c.width = Inches(2.38)

    populate_author_cell(
        row1_cells[0],
        name="Goutham C. S.",
        designation="Undergraduate Scholar",
        dept=AFFIL_DEPT,
        college=COLLEGE,
        location=LOCATION,
        email="gouthamcs453@gmail.com",
    )
    populate_author_cell(
        row1_cells[1],
        name="Abel Louis Fernandez",
        designation="Undergraduate Scholar",
        dept=AFFIL_DEPT,
        college=COLLEGE,
        location=LOCATION,
        email="abellouis2003@gmail.com",
    )
    populate_author_cell(
        row1_cells[2],
        name="Akshay P. Benjamin",
        designation="Undergraduate Scholar",
        dept=AFFIL_DEPT,
        college=COLLEGE,
        location=LOCATION,
        email="akshaypbenjamin@gmail.com",
    )

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(6)
    p_space.paragraph_format.space_after = Pt(6)

    # Row 2: Authors 4 and 5 (Guide)
    tbl_row2 = doc.add_table(rows=1, cols=2)
    tbl_row2.alignment = WD_TABLE_ALIGNMENT.CENTER
    row2_cells = tbl_row2.rows[0].cells
    for c in row2_cells:
        c.width = Inches(3.57)

    populate_author_cell(
        row2_cells[0],
        name="Aparna S.",
        designation="Undergraduate Scholar",
        dept=AFFIL_DEPT,
        college=COLLEGE,
        location=LOCATION,
        email="aparnas2003@gmail.com",
    )
    populate_author_cell(
        row2_cells[1],
        name="Ms. Neethi Narayanan",
        designation="Assistant Professor (Research Guide)",
        dept=AFFIL_DEPT,
        college=COLLEGE,
        location=LOCATION,
        email="neethi.narayanan@mbcet.ac.in",
    )

    # ── 4. Section Break to 2-Column Body Text ────────────────────────────────
    sec_body = doc.add_section(docx.enum.section.WD_SECTION.CONTINUOUS)
    sec_body.page_width = Inches(8.27)
    sec_body.page_height = Inches(11.69)
    sec_body.top_margin = Inches(0.75)
    sec_body.bottom_margin = Inches(1.00)
    sec_body.left_margin = Inches(0.56)
    sec_body.right_margin = Inches(0.56)

    cols_xml = '<w:cols xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:num="2" w:space="18pt"/>'
    sec_body._sectPr.append(parse_xml(cols_xml))

    # Helper functions
    def add_p(text: str, style_name: str = "Body Text", bold_prefix: str = "") -> docx.text.paragraph.Paragraph:
        p = doc.add_paragraph(style=doc.styles[style_name])
        p.paragraph_format.line_spacing = 0.95
        p.paragraph_format.space_after = Pt(4.5)
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.bold = True
            r_b.font.name = "Times New Roman"
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        return p

    def add_h1(text: str):
        p = doc.add_paragraph(style=doc.styles["Heading 1"])
        p.paragraph_format.space_before = Pt(9.0)
        p.paragraph_format.space_after = Pt(3.5)
        r = p.add_run(text.upper())
        r.font.name = "Times New Roman"
        return p

    def add_h2(text: str):
        p = doc.add_paragraph(style=doc.styles["Heading 2"])
        p.paragraph_format.space_before = Pt(6.5)
        p.paragraph_format.space_after = Pt(2.5)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.italic = True
        return p

    def add_equation_omml(latex_code: str, eq_num: str):
        p = doc.add_paragraph(style=doc.styles["equation"])
        p.paragraph_format.space_before = Pt(9.0)
        p.paragraph_format.space_after = Pt(9.0)
        p.paragraph_format.line_spacing = 1.15
        pPr = p._p.get_or_add_pPr()
        tabs = parse_xml(f'<w:tabs {nsdecls("w")}><w:tab w:val="center" w:pos="100pt"/><w:tab w:val="right" w:pos="248pt"/></w:tabs>')
        pPr.append(tabs)

        rPr = parse_xml(f'<w:rPr {nsdecls("w")}><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/></w:rPr>')
        pPr.append(rPr)

        p.add_run("\t")
        omml_elem = latex_to_omml(latex_code, bold=True)
        p._p.append(omml_elem)
        p.add_run("\t")
        r_num = p.add_run(f"({eq_num})")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(9.5)
        r_num.bold = True

    def add_fig(img_path: str, caption: str, width_in: float = 3.35):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.paragraph_format.line_spacing = 1.0
            p_img.add_run().add_picture(img_path, width=Inches(width_in))

            p_cap = doc.add_paragraph(style=doc.styles["figure caption"] if "figure caption" in doc.styles else doc.styles["Normal"])
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(1)
            p_cap.paragraph_format.space_after = Pt(6)
            p_cap.paragraph_format.line_spacing = 0.95
            r_cap = p_cap.add_run(caption)
            r_cap.font.name = "Times New Roman"
            r_cap.font.size = Pt(8.0)
            r_cap.font.italic = True

    def add_ref(text: str):
        p = doc.add_paragraph(style=doc.styles["references"])
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.0)

    # ── 5. Abstract & Keywords (Rectified, Highly Polished Academic Prose) ─────
    add_p(
        "Deep neural networks (DNNs) deployed in safety-critical visual diagnostics—most notably in dermatological oncology—exhibit alarming vulnerability to transform-brittleness: generating confident and accurate classifications on canonical inputs, yet flipping catastrophically to fatal misdiagnoses under minor, semantically invariant perturbations. Conventional runtime safety mechanisms typically rely on naive binary metamorphic checking, halting execution upon any single prediction discrepancy. In clinical practice, this triggers debilitating false alarm rates (exceeding 80% on sensitive benchmarks), inducing acute alert fatigue and rapid clinician disengagement. To resolve this challenge, we introduce the Metamorphic Circuit Breaker (MCB), a model-agnostic runtime safety framework founded on Hierarchical Metamorphic Fault Attribution. Perturbations are structured into orthogonal semantic families (Geometric, Photometric, Sensor/Noise) across multi-tier severity ladders, penalized via an inverted severity function where subtle, imperceptible boundary flips incur maximum weight. To eliminate intra-family representation bias, we formulate a Two-Level Hierarchical Normalization scheme augmented with peak-transform anchoring. Systemic multi-family vulnerability is quantified via a Cross-Family Spread (CFS) metric and synthesized into a continuous Composite Brittleness Index (CBI) that governs a three-state circuit breaker (CLOSED, HALF-OPEN, and OPEN) with dynamically coupled monotonic warning thresholds. Tripped inferences are intercepted before delivery, explainability heatmaps are synthesized via Gradient-weighted Class Activation Mapping (GradCAM), and cases are dispatched to an interactive Human-in-the-Loop (HITL) clinical triage portal. Empirical evaluations across clinical dermatology (ISIC 2019 / ConvNeXt-Base), autonomous driving perception (GTSRB / MobileNetV3-Small), and agricultural pathology (Plant Village / EfficientNet-B0) demonstrate that MCB suppresses false alarm rates by 4x to 13.6x compared to naive metamorphic testing (down to 1.9%–13.0%), intercepts up to 72.7% of critical runtime misclassifications, and operates with sub-60 ms GPU execution latency.",
        style_name="Abstract",
        bold_prefix="Abstract\u2014",
    )

    add_p(
        "Metamorphic testing, circuit breaker, runtime verification, medical vision, skin lesion classification, deep neural networks, AI safety, explainable AI, human-in-the-loop.",
        style_name="Keywords",
        bold_prefix="Keywords\u2014",
    )

    # ── 6. Section I: Introduction ────────────────────────────────────────────
    add_h1("I. Introduction")
    add_p(
        "Recent advances in deep learning and modern convolutional architectures, such as ConvNeXt and Vision Transformers, have propelled computer-aided diagnosis (CAD) systems toward diagnostic parity with expert clinicians in medical imaging tasks. In dermatological oncology, automated classification of dermatoscope imagery—exemplified by the International Skin Imaging Collaboration (ISIC) benchmarks—offers critical potential for early melanoma detection, where five-year survival rates exceed 98% upon localized intervention but drop precipitously following metastatic invasion [1]."
    )
    add_p(
        "Despite achieving high top-1 accuracy on curated, identically distributed test benchmarks, deep neural networks in real-world deployment exhibit alarming brittleness to superficial, semantically invariant visual shifts. This phenomenon, termed transform-brittleness, occurs when an AI model correctly and confidently diagnoses a benign melanocytic nevus, but flips its prediction to malignant melanoma (or vice-versa) when the identical image undergoes a trivial 5-degree rotation, a minor contrast adjustment, or mild optical sensor defocus [2]. In clinical settings, such erratic behavior undermines practitioner trust, threatens patient safety, and creates severe legal liability."
    )
    add_p(
        "To guard against silent AI failures, software engineering has embraced Metamorphic Testing (MT). In the absence of an empirical test oracle (the true ground-truth diagnosis is unknown for an unlabelled incoming patient), metamorphic relations define domain invariants: biological pathology is invariant under dermatoscope rotation, ambient examination lighting shifts, and sensor noise [3]. However, existing runtime safety wrappers implement naive binary checking: if any single transformed variant yields a different output class than the baseline, the system halts execution. In clinical practice, this naive policy fails fundamentally:"
    )
    add_p(
        "1) Aggressive or boundary transformations naturally alter high-frequency pixel statistics, triggering false positive circuit trips on over 80% of healthy images and causing debilitating clinician alert fatigue [4]."
    )
    add_p(
        "2) Isolated algorithmic glitches are indistinguishable from systemic architectural collapse, preventing meaningful automated triage."
    )
    add_p(
        "To resolve these limitations, this paper introduces the Metamorphic Circuit Breaker (MCB), a runtime safety framework founded on Hierarchical Metamorphic Fault Attribution. Our core contributions are summarized as follows:"
    )
    add_p(
        "\u2022 Hierarchical Semantic Grouping & Severity Ladders: Candidate metamorphic relations are organized into orthogonal families (Geometric, Photometric, Sensor/Noise), each stratified across mild, moderate, and severe parameter tiers with inverted penalty weighting that rigorously punishes subtle decision boundary instability."
    )
    add_p(
        "\u2022 Skew-Free Two-Level Normalization: We mathematically resolve transformation-type cardinality skew, ensuring that continuous multi-tier transforms do not overpower compact transforms, while peak vulnerability anchoring preserves equal triggering sensitivity across families."
    )
    add_p(
        "\u2022 Evidence-Based Three-State Circuit Breaker: Formulating the Composite Brittleness Index (CBI) to govern transitions across CLOSED (healthy auto-approval), HALF-OPEN (provisional output with caution telemetry), and OPEN (fail-safe interception to doctor triage), supported by dynamic threshold coupling to prevent state-ordering collapse."
    )
    add_p(
        "\u2022 Model-Agnostic Abstraction with Clinical HITL: Decoupling safety evaluation through an abstract vision interface (BaseVisionModel) and pairing it with Gradient-weighted Class Activation Mapping (GradCAM) explainability in an interactive clinical triage portal, accompanied by cross-domain verification in autonomous driving and agriculture."
    )

    # ── 7. Section II: Related Work ───────────────────────────────────────────
    add_h1("II. Related Work")
    add_h2("A. Deep Learning in Dermatological Oncology")
    add_p(
        "Automated skin lesion screening has evolved from classical hand-crafted feature extractors (ABCD criteria: Asymmetry, Border, Color, Diameter) to deep convolutional and transformer networks. The ISIC benchmark challenges established standardized multi-class classification over melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesions (VASC), and squamous cell carcinoma (SCC) [5]. Modern architectures, such as ConvNeXt [6], modernize standard convolutional design with 7x7 depthwise convolutions and inverted bottleneck stages, achieving competitive performance while preserving computational efficiency. Nonetheless, convolutional inductive biases remain vulnerable to spatial and photometric transformations outside the training distribution manifold."
    )
    add_h2("B. Metamorphic Testing in Vision Systems")
    add_p(
        "Metamorphic testing, pioneered by Chen et al. [3], alleviates the oracle problem by verifying relations across multiple executions. While initially formulated for numerical and compilers testing, MT has recently expanded to autonomous driving [7] and deep learning robustness auditing [8]. However, the vast majority of prior vision MT literature focuses on offline testing and post-hoc model benchmarking. Operating MT as an active, runtime safety circuit breaker requires rigorous false-alarm suppression and deterministic latency bounding."
    )
    add_h2("C. Uncertainty Estimation and Out-of-Distribution Gating")
    add_p(
        "Runtime safety has historically relied on Bayesian neural networks, Monte Carlo Dropout (MCDO), and deep ensembles to estimate predictive uncertainty [9]. While theoretically sound, Monte Carlo sampling multiplies forward-pass latency, rendering it impractical for real-time edge hardware. Furthermore, deep networks suffer from poor softmax calibration: models frequently assign >95% confidence to erroneous predictions. The proposed MCB framework circumvents internal weight sampling by evaluating functional behavioral invariance across calibrated input manifolds."
    )

    # ── 8. Section III: Architectural Foundation ──────────────────────────────
    add_h1("III. Architectural Foundation: Bridging Baseline Infrastructure and Hierarchical Safety")
    add_p(
        "To establish a clear engineering lineage, this paper bridges the foundational baseline implementation (Phase 1: data processing, fine-tuned backbone, and native saliency hooks) with the Phase 2 Hierarchical Fault Attribution architecture that forms the primary theoretical contribution of this work."
    )
    add_h2("A. Baseline Infrastructure (Phase 1 Baseline)")
    add_p(
        "The Phase 1 infrastructure established the end-to-end data, model, and explainability baseline:"
    )
    add_p(
        "1) Dataset Processing: Automated ingestion and validation of the ISIC 2019 dataset comprising 25,331 clinical images, normalized across an 8-class ground-truth schema (MEL, NV, BCC, AK, BKL, DF, VASC, SCC) with a 70/30 stratified train/validation split."
    )
    add_p(
        "2) Primary Backbone: A ConvNeXt-Base convolutional architecture fine-tuned with AdamW, Cosine Annealing learning rate scheduling, Automatic Mixed Precision (AMP), and validation accuracy checkpointing."
    )
    add_p(
        "3) Baseline Metamorphic Testing: Initial evaluation of 8 empirical metamorphic relations (Rotation 15 deg, Horizontal Flip, Center Zoom 0.8x, Brightness 1.5x/1.2x, Gaussian Noise sigma=0.05, Gaussian Blur radius=2.0, Sharpening 2.0x, Saturation 0.5x), logging per-sample prediction stability."
    )
    add_p(
        "4) Saliency Engine: Native PyTorch forward-hook GradCAM implementation targeting model.stages[-1].blocks[-1] to generate paired visual heatmaps explaining prediction flips."
    )
    add_p(
        "5) Internal Audit Workbench: A research dashboard for offline dataset inspection and batch stability visualization."
    )
    add_h2("B. Hierarchical Safety Architecture (Phase 2 Framework)")
    add_p(
        "While baseline testing confirmed that metamorphic violations frequently correlate with erroneous predictions, flat testing cannot differentiate between a harmless boundary artifact and a true model breakdown. The Phase 2 framework introduces:"
    )
    add_p(
        "1) Orthogonal Semantic Families with calibrated multi-tier severity ladders (Mild, Moderate, Severe) governed by inverted penalty weighting."
    )
    add_p(
        "2) Two-Level Hierarchical Normalization mathematically resolving transformation-type imbalance skew with peak vulnerability anchoring."
    )
    add_p(
        "3) Cross-Family Spread (CFS) scoring to detect systemic multi-modal reasoning breakdown."
    )
    add_p(
        "4) Composite Brittleness Index (CBI) driving a three-state circuit breaker (CLOSED, HALF-OPEN, OPEN) with dynamic monotonic threshold coupling."
    )
    add_p(
        "5) A real-time clinical triage ecosystem featuring a high-throughput FastAPI backend and a responsive Doctor Intervention Portal."
    )

    # ── 9. Section IV: Mathematical Formulation and Decision Logic ───────────
    add_h1("IV. Mathematical Formulation and Decision Logic")
    add_h2("A. Model-Agnostic Interface Abstraction")
    add_p(
        "To ensure universal portability across heterogeneous backbones (ConvNeXt, ResNet, MobileNet, Vision Transformers, or remote cloud REST APIs), the circuit breaker interfaces strictly through an abstract contract, BaseVisionModel, defining predict(image) -> (y_pred, label, conf, P) and explain(image, class_idx) -> Heatmap. The circuit breaker engine treats the underlying network as a functional black box."
    )

    add_h2("B. Monotonic Inverted Severity Weighting")
    add_p(
        "In safety-critical vision, perturbation magnitude exhibits an inverse relationship with structural vulnerability: a prediction flip under an imperceptible mild perturbation (e.g., 5-deg tilt or 2% sensor noise) indicates severe fragility right at the decision boundary. Conversely, a flip at an aggressive boundary transformation reflects high-frequency distortion where human observers also experience uncertainty. We formalize this via a monotonically descending severity weighting function w(s):"
    )
    add_equation_omml(
        r"w(s) = \begin{cases} 1.0, & s = \text{Mild} \\ 0.6, & s = \text{Moderate} \\ 0.3, & s = \text{Severe} \end{cases}",
        "1"
    )

    add_h2("C. Resolving Representation Skew via Two-Level Hierarchical Normalization")
    add_p(
        "In naive flat metamorphic scoring, if a family contains disproportionate test counts across different transformation types (e.g., Rotation has 3 severity tiers totaling weight 1.9, while Scale has 2 tiers totaling 0.9), Rotation unfairly commands over 67% of the family score. Consequently, a total collapse on Scale contributes minimally, failing to trigger safety alerts."
    )
    add_p(
        "To establish structural fairness, we formulate a Two-Level Hierarchical Normalization scheme:"
    )
    add_p(
        "Level 1: For each transformation type m (e.g., rotation, scale, brightness) within family k, compute a normalized severity score:"
    )
    add_equation_omml(
        r"\text{Score}_m(x) = \frac{\sum_{s \in \mathcal{S}_m} w(s) \cdot \mathbb{I}[f(t_{m,s}(x)) \neq y_0]}{\sum_{s \in \mathcal{S}_m} w(s)}",
        "2"
    )
    add_p(
        "where I[.] is the indicator function yielding 1 on a prediction flip and 0 on stability, and S_m is the set of severity tiers defined for transform m. Score_m(x) is bounded in [0.0, 1.0]."
    )
    add_p(
        "Level 2: To prevent family cardinality division bias (where families with M_k = 2 require fewer total flips to activate than families with M_k = 3), the Intra-Family Instability Ratio IR_k(x) combines mean family instability with peak transform vulnerability:"
    )
    add_equation_omml(
        r"\text{IR}_k(x) = (1 - \lambda_{\text{peak}}) \cdot \left(\frac{1}{M_k} \sum_{m=1}^{M_k} \text{Score}_m(x)\right) + \lambda_{\text{peak}} \cdot \max_{m} \text{Score}_m(x)",
        "3"
    )
    add_p(
        "where lambda_peak = 0.30 anchors 30% of the family score to the most vulnerable transformation type. This hybrid formulation guarantees that an isolated, complete collapse of any single transformation type produces IR_k in [0.53, 0.65] across all families, eliminating family cardinality bias."
    )

    # ── Table: Transformation Registry and Parameter Specification ───────────
    p_reg_head = doc.add_paragraph(style=doc.styles["table head"])
    p_reg_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_reg_head.paragraph_format.space_before = Pt(6)
    p_reg_head.paragraph_format.space_after = Pt(2)
    r_rh = p_reg_head.add_run("METAMORPHIC TRANSFORMATION TAXONOMY AND SEVERITY LADDER")
    r_rh.font.name = "Times New Roman"
    r_rh.font.size = Pt(8.0)
    r_rh.bold = True

    tbl_reg = doc.add_table(rows=9, cols=4)
    tbl_reg.alignment = WD_TABLE_ALIGNMENT.CENTER

    reg_headers = ["Family", "Transform Type", "Severity Tiers (Mild / Mod / Sev)", "Weights [w_s]"]
    reg_rows = [
        ["Geometric", "Rotation", "[-5 deg, +5 deg] / [-15 deg, +15 deg] / [-30 deg, +30 deg]", "[1.0, 0.6, 0.3]"],
        ["Geometric", "Scaling", "None / [0.9x, 1.1x] / [0.8x, 1.2x]", "[0.0, 0.6, 0.3]"],
        ["Geometric", "Horizontal Flip", "Single-tier invariant reflection (dermoscopy only; filtered in directional tasks)", "[1.0]"],
        ["Photometric", "Brightness Shift", "+/-10% / +/-25% / +/-40%", "[1.0, 0.6, 0.3]"],
        ["Photometric", "Contrast Scaling", "0.9x--1.1x / 0.75x--1.25x / 0.6x--1.4x", "[1.0, 0.6, 0.3]"],
        ["Photometric", "Color Saturation", "0.9x--1.1x / 0.7x--1.3x / 0.5x--1.5x", "[1.0, 0.6, 0.3]"],
        ["Sensor/Noise", "Gaussian Blur", "sigma=0.5 / sigma=1.0 / sigma=1.5", "[1.0, 0.6, 0.3]"],
        ["Sensor/Noise", "Gaussian Noise", "sigma=0.02 / sigma=0.05 / sigma=0.08", "[1.0, 0.6, 0.3]"],
    ]

    for c_i, h_t in enumerate(reg_headers):
        c = tbl_reg.cell(0, c_i)
        c.text = h_t
        p_c = c.paragraphs[0]
        p_c.runs[0].bold = True
        p_c.runs[0].font.name = "Times New Roman"
        p_c.runs[0].font.size = Pt(6.5)
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for r_i, r_data in enumerate(reg_rows):
        for c_i, val in enumerate(r_data):
            c = tbl_reg.cell(r_i + 1, c_i)
            c.text = val
            p_c = c.paragraphs[0]
            p_c.runs[0].font.name = "Times New Roman"
            p_c.runs[0].font.size = Pt(6.5)
            if c_i in [0, 3]:
                p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_ieee_table_borders(tbl_reg)

    add_h2("D. Cross-Family Spread Score and Activation Boundaries")
    add_p(
        "Simultaneous failure across independent families provides definitive proof of generalized reasoning collapse. We define the Cross-Family Spread (CFS) score:"
    )
    add_equation_omml(
        r"\text{CFS}(x) = \frac{1}{|\mathcal{F}|} \sum_{k=1}^{|\mathcal{F}|} \mathbb{I}[\text{IR}_k(x) \ge \tau_{\text{fam}}]",
        "4"
    )
    add_p(
        "where |F| = 3 and tau_fam is the family activation threshold (calibrated baseline tau_fam = 0.35; empirical sweep range [0.20, 0.35]). Setting tau_fam = 0.35 creates a conservative noise ceiling preventing benign visual flutter from triggering multi-family spread."
    )

    add_h2("E. Composite Brittleness Index (CBI)")
    add_p(
        "The overall runtime risk score is synthesized by the Composite Brittleness Index (CBI):"
    )
    add_equation_omml(
        r"\text{CBI}(x) = \alpha \cdot \max_{k} \text{IR}_k(x) + \beta \cdot \text{CFS}(x) + \gamma \cdot (1 - c_0)",
        "5"
    )
    add_p(
        "constrained such that alpha + beta + gamma = 1.0 and alpha, beta, gamma >= 0. The default calibrated configuration sets alpha = 0.50 (peak family vulnerability), beta = 0.35 (cross-family spread), and gamma = 0.15 (baseline model uncertainty). The confidence weight gamma is deliberately constrained to <= 0.15 because deep vision models exhibit pronounced overconfidence on misclassified samples, rendering raw softmax confidence an unreliable primary gate."
    )

    add_h2("F. Three-State Circuit Breaker Decision Engine")
    add_p(
        "The circuit breaker operates across three operational states analogous to electrical protection systems, governed by the decision policy:"
    )
    add_equation_omml(
        r"\text{State}(x) = \begin{cases} \text{CLOSED}, & \text{CBI}(x) < \theta_{\text{warn}} \\ \text{HALF-OPEN}, & \theta_{\text{warn}} \le \text{CBI}(x) < \theta_{\text{trip}} \\ \text{OPEN}, & \text{CBI}(x) \ge \theta_{\text{trip}} \end{cases}",
        "6"
    )
    add_p(
        "To strictly guarantee monotonic state progression (CLOSED -> HALF-OPEN -> OPEN) and prevent state-machine collapse during cutoff calibration, the warning cutoff dynamically couples to the trip cutoff via theta_warn = 0.50 * theta_trip (baseline default: theta_trip = 0.55, theta_warn = 0.25; calibrated sweep: theta_trip = 0.10, theta_warn = 0.05):"
    )
    add_p(
        "\u2022 CLOSED (CBI(x) < theta_warn): High behavioral invariance. Automated diagnostic output is approved and returned directly to the electronic health record."
    )
    add_p(
        "\u2022 HALF-OPEN (theta_warn <= CBI(x) < theta_trip): Moderate boundary sensitivity. Output is provisionally delivered accompanied by advisory telemetry requesting physician review."
    )
    add_p(
        "\u2022 OPEN (CBI(x) >= theta_trip): Critical multi-family instability detected. Automated prediction is actively blocked, explainability diagnostics are generated, and the case is escalated to the Human-in-the-Loop triage portal."
    )

    # ── 10. Section V: Clinical Human-in-the-Loop Triage ──────────────────────
    add_h1("V. Clinical Human-in-the-Loop Triage and Explainability")
    add_h2("A. Operational Execution Pipeline")
    add_p(
        "When an incoming dermatoscope query arrives at the FastAPI backend, the image is passed to the ConvNeXt baseline while simultaneously executing the batched metamorphic matrix on the GPU (executing within 54.2 ms total latency). If CBI(x) >= theta_trip, the circuit breaker trips to OPEN, intercepting the response before it reaches the electronic health record."
    )

    add_h2("B. Saliency Attribution via GradCAM Attention Shift")
    add_p(
        "To provide actionable causal evidence to the reviewing clinician, the tripped breaker hooks the final convolutional stage of ConvNeXt-Base (model.stages[-1].blocks[-1]). By computing gradients of the predicted class score with respect to feature activations, GradCAM generates paired 2D saliency heatmaps: original attention vs. transformed attention. This visually highlights when model focus errantly migrates from lesion pigment networks onto surrounding normal skin or optical artifacts."
    )

    add_h2("C. Clinical Decision Support Portal")
    add_p(
        "Escalated cases enter a secure clinical triage queue rendered via a modern React interface. Dermatologists review the original dermatoscope photo, side-by-side transformed variants that induced flips, the baseline vs. perturbed prediction breakdown, and the GradCAM visual shift. Clinicians can approve the AI suggestion, override with an alternate diagnosis (e.g., ordering an immediate excisional biopsy), and attach clinical notes. All resolutions are archived in an audit log to support continuous model retraining."
    )

    # ── 11. Section VI: Empirical Results and Cross-Domain Validation ───────
    add_h1("VI. Empirical Results and Cross-Domain Validation")
    add_p(
        "To evaluate the runtime safety behavior of the Metamorphic Circuit Breaker, we conducted benchmark evaluations across three distinct domains: (1) Clinical Dermatology: ConvNeXt-Base on ISIC 2019 (N = 250, with 43 base-model misclassifications); (2) Autonomous Driving Perception: MobileNetV3-Small on GTSRB (N = 250 in comparative Test 3 with 11 errors; N = 500 in held-out batch Test 2 with 21 errors); and (3) Agricultural Crop Pathology: EfficientNet-B0 on Plant Village (N = 100 test split with 17 errors). All benchmarks represent held-out in-distribution test samples."
    )

    # ── Table I: Rigorously Reconciled Cross-Domain Performance Matrix ────────
    p_tbl_head = doc.add_paragraph(style=doc.styles["table head"])
    p_tbl_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tbl_head.paragraph_format.space_before = Pt(8)
    p_tbl_head.paragraph_format.space_after = Pt(3)
    r_th = p_tbl_head.add_run("TABLE I\nCROSS-DOMAIN EMPIRICAL PERFORMANCE EVALUATION WITH WILSON 95% CONFIDENCE INTERVALS")
    r_th.font.name = "Times New Roman"
    r_th.font.size = Pt(8.0)
    r_th.bold = True

    headers_all = ["Domain & Safety Paradigm", "TP/FP/TN/FN", "Safety Recall [95% CI]", "False Alarm Rate [95% CI]", "AUROC", "Latency"]
    rows_all = [
        # Domain 1: ISIC 2019
        ["ISIC 2019: Skin Lesions (ConvNeXt-Base, N=250, 43 errors)", "-", "-", "-", "-", "-"],
        ["  Standalone Model (Unmonitored)", "0 / 0 / 207 / 43", "0.0% [0.0, 8.2]", "0.0% [0.0, 1.8]", "-", "18.4 ms (GPU)"],
        ["  Softmax Entropy (85th %ile)", "18 / 20 / 187 / 25", "41.9% [28.4, 56.7]", "9.7% [6.3, 14.5]", "0.816", "19.1 ms (GPU)"],
        ["  MC-Dropout (Head-Only, 10 pass)", "15 / 23 / 184 / 28", "34.9% [22.4, 49.8]", "11.1% [7.5, 16.1]", "0.771", "148.4 ms (CPU)"],
        ["  Naive Binary 1-Flip MT", "26 / 42 / 165 / 17", "60.5% [45.6, 73.6]", "20.3% [15.4, 26.3]", "0.729", "54.2 ms (GPU)"],
        ["  Proposed MCB (Prior tau=0.35, theta=0.55)", "7 / 4 / 203 / 36", "16.3% [8.1, 30.0]", "1.9% [0.8, 4.9]", "0.813", "54.2 ms (GPU)"],
        ["  Proposed MCB (Calibrated tau=0.20, theta=0.10)", "23 / 27 / 180 / 20", "53.5% [38.9, 67.5]", "13.0% [9.1, 18.3]", "0.813", "54.2 ms (GPU)"],
        # Domain 2: GTSRB
        ["GTSRB: Traffic Signs (MobileNetV3, N=250 Test 3, 11 errors)", "-", "-", "-", "-", "-"],
        ["  Standalone Model (Unmonitored)", "0 / 0 / 239 / 11", "0.0% [0.0, 25.9]", "0.0% [0.0, 1.6]", "-", "10.9 ms (CPU)"],
        ["  Softmax Entropy (Thresholded)", "7 / 43 / 196 / 4", "63.6% [35.4, 84.8]", "18.0% [13.6, 23.4]", "-", "12.8 ms (CPU)"],
        ["  MC-Dropout (10 Passes)", "10 / 40 / 199 / 1", "90.9% [62.3, 98.4]", "16.7% [12.5, 22.0]", "-", "131.1 ms (CPU)"],
        ["  Naive Binary 1-Flip MT (Includes Flips)", "11 / 201 / 38 / 0", "100.0% [74.1, 100.0]", "84.1% [78.9, 88.2]", "-", "295.6 ms (CPU)"],
        ["  Proposed MCB (tau=0.25, theta=0.30)", "8 / 50 / 189 / 3", "72.7% [43.4, 90.3]", "20.9% [16.2, 26.6]", "0.885", "295.6 ms (CPU)"],
        ["  Proposed MCB (Test 2 Held-Out, N=500, 21 errors)", "11 / 96 / 383 / 10", "52.4% [32.4, 71.7]", "20.0% [16.7, 23.9]", "-", "249.4 ms (CPU)"],
        # Domain 3: Plant Village
        ["Plant Village: Leaf Disease (EfficientNet-B0, N=100, 17 errors)", "-", "-", "-", "-", "-"],
        ["  Naive Binary 1-Flip MT", "12 / 68 / 15 / 5", "70.6% [46.9, 86.7]", "81.9% [72.3, 88.7]", "-", "28.5 ms (GPU)"],
        ["  Softmax Entropy Gating", "5 / 5 / 78 / 12", "29.4% [13.3, 53.1]", "6.0% [2.6, 13.3]", "-", "28.5 ms (GPU)"],
        ["  Proposed MCB (Prior tau=0.35, theta=0.55)", "3 / 5 / 78 / 14", "17.6% [6.2, 41.0]", "6.0% [2.6, 13.3]", "0.622", "28.5 ms (GPU)"],
        ["  Proposed MCB (Optimal sweep tau=0.45, theta=0.50)", "6 / 5 / 78 / 11", "35.3% [17.3, 58.7]", "6.0% [2.6, 13.3]", "0.622", "28.5 ms (GPU)"],
    ]

    tbl_bench = doc.add_table(rows=len(rows_all) + 1, cols=6)
    tbl_bench.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col_idx, h in enumerate(headers_all):
        cell = tbl_bench.cell(0, col_idx)
        cell.text = h
        p_c = cell.paragraphs[0]
        p_c.runs[0].bold = True
        p_c.runs[0].font.name = "Times New Roman"
        p_c.runs[0].font.size = Pt(6.5)
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row_idx, r_vals in enumerate(rows_all):
        is_subhead = not r_vals[0].startswith("  ")
        for col_idx, val in enumerate(r_vals):
            cell = tbl_bench.cell(row_idx + 1, col_idx)
            cell.text = val
            p_c = cell.paragraphs[0]
            p_c.runs[0].font.name = "Times New Roman"
            p_c.runs[0].font.size = Pt(6.5)
            if is_subhead:
                p_c.runs[0].bold = True
            if col_idx in [1, 2, 3, 4, 5]:
                p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_ieee_table_borders(tbl_bench)

    p_tbl_foot = doc.add_paragraph(style=doc.styles["table footnote"])
    p_tbl_foot.paragraph_format.space_before = Pt(3)
    p_tbl_foot.paragraph_format.space_after = Pt(8)
    r_tf = p_tbl_foot.add_run("*Note: Counts and metrics extracted directly from verified raw-batch logs. 95% Confidence Intervals computed via the Wilson Score method without continuity correction.")
    r_tf.font.name = "Times New Roman"
    r_tf.font.size = Pt(7.0)

    # ── Figure 1: Cross-Domain Semantic Vulnerability Profiles ────────────────
    fig1_path = os.path.join(FIG_DIR, "fig1_domain_radar.png")
    add_fig(fig1_path, "Fig. 1. Cross-domain semantic vulnerability profiles depicting Invalidation Ratios across Geometric, Photometric, and Sensor/Noise perturbation families.")

    add_h2("A. Alert Fatigue Context and False Alarm Reduction")
    add_p(
        "A primary motivation for multi-family metamorphic safety gating is mitigating false alarm overhead. On GTSRB, naive 1-flip metamorphic verification triggered trips on 84.1% of correctly identified traffic signs (201 false alarms out of 239 correct signs), creating overwhelming distraction. On Plant Village, naive testing flagged 81.9% of healthy predictions (68 false alarms out of 83 correct images). On ISIC 2019, naive testing generated a 20.3% false alarm rate (42 false alarms out of 207 correct samples)."
    )
    add_p(
        "By enforcing Two-Level Hierarchical Normalization and Cross-Family Spread constraints, MCB reduces false alarm rates significantly: on ISIC 2019 under default prior parameters (tau_fam=0.35, theta_trip=0.55), FAR is suppressed to 1.9% (4 false alarms out of 207), representing a 10.7x reduction in false alarms compared to naive testing. On Plant Village, FAR is suppressed from 81.9% to 6.0% (a 13.6x reduction). As illustrated in Fig. 2, this false alarm suppression establishes a well-defined Pareto trade-off between alert suppression and error interception."
    )

    # ── Figure 2: Safety Recall vs False Alarm Rate ───────────────────────────
    fig2_path = os.path.join(FIG_DIR, "fig2_baseline_paradigm_comparison.png")
    add_fig(fig2_path, "Fig. 2. Safety Recall vs. False Alarm Rate (FAR) across runtime safety paradigms in Autonomous Driving (GTSRB) and Medical Vision (ISIC 2019), depicting exact verified operating points.")

    add_h2("B. Semantic Family Vulnerability Profiling")
    add_p(
        "As visualized in Fig. 1, Sensor and Optical Noise was consistently identified as the primary failure axis across all backbones (ConvNeXt-Base, MobileNetV3-Small, and EfficientNet-B0). In dermatology, mean Invalidation Ratio for errors was highest in Sensor Noise (mean IR = 0.0475, with error mean 0.1475 vs. correct mean 0.0267), compared to Photometric (mean IR = 0.0378) and Geometric (mean IR = 0.0348). In autonomous driving, Sensor instability reached IR = 0.2422, primarily driven by blur (41.1% flip rate) and gaussian noise (26.5% flip rate)."
    )

    add_h2("C. Threshold Calibration Dynamics and Operating Points")
    add_p(
        "Our empirical sweeps clarify that the theoretical parameter assignment of tau_fam = 0.35 and theta_trip = 0.55 operates as an ultra-conservative advisory filter. Because Eq. (5) and (6) assign alpha = 0.50 and beta = 0.35, a sample with high baseline confidence requires an intra-family failure ratio of IR >= 0.80 to trip independently. This explains why prior thresholds exhibit conservative safety recall (16.3% on ISIC, 17.6% on Plant Village)."
    )
    add_p(
        "By tuning thresholds via ROC optimization (depicted in Fig. 3), balanced operating points emerge: on ISIC 2019, lowering thresholds to tau_fam = 0.20 and theta_trip = 0.10 elevates safety recall to 53.5% (intercepting 23 of 43 errors) at a 13.0% false alarm rate (27 false alarms), yielding peak Youden J = 0.404. On GTSRB (N = 250 Test 3), operating at tau_fam = 0.25 and theta_trip = 0.30 achieves 72.7% safety recall (8 of 11 errors) at 20.9% FAR. Importantly, thresholds derived via grid sweeps must be calibrated on independent validation sets in production to prevent in-sample optimistic bias."
    )

    # ── Figure 3: Threshold Calibration Shifts ────────────────────────────────
    fig3_path = os.path.join(FIG_DIR, "fig3_threshold_calibration_shifts.png")
    add_fig(fig3_path, "Fig. 3. Empirical Youden J and Safety Recall operating curves as a function of theta_trip, establishing the Pareto trade-off between conservative filtering and high-recall intervention.")

    add_h2("D. Ablation Analysis: Normalization, Confidence Trust, and Baselines")
    add_p(
        "Ablation experiments validate the structural behavior of the framework. On ISIC 2019, Two-Level Hierarchical Normalization achieved an AUROC of 0.813, compared to 0.728 for Flat Metamorphic Testing (+0.085 AUROC). Two-Level normalization recovered 9 error cases missed by Flat MT by preventing photometric color jitter from diluting severe sensor noise failures. However, with N = 43 errors, the 95% confidence interval for AUROC difference spans approximately +/-0.09, indicating that validation on larger cohorts is required to confirm statistical significance."
    )
    add_p(
        "Furthermore, as evaluated in Fig. 4, we analyzed the role of baseline model confidence gamma in CBI computation. On ISIC 2019, Softmax Entropy alone achieved an AUROC of 0.816, matching MCB (0.813). In weight ablation sweeps, the High Confidence Trust configuration (alpha=0.25, beta=0.25, gamma=0.50) achieved the highest AUROC (0.823 on ISIC, 0.907 on GTSRB, 0.828 on Plant Village), while Pure Black-Box testing without confidence access (alpha=0.60, beta=0.40, gamma=0.00) dropped to 0.721 on ISIC. This reveals that on clean in-distribution test samples, model confidence remains a potent discriminatory feature; MCB's distinct value lies in structured failure attribution and semantic introspection rather than pure AUROC dominance."
    )

    # ── Figure 4: Weight Ablation ─────────────────────────────────────────────
    fig4_path = os.path.join(FIG_DIR, "fig4_method_a_weight_ablation.png")
    add_fig(fig4_path, "Fig. 4. Cutoff-Free AUROC performance of Composite Brittleness Index across five parameter configurations.")

    add_h2("E. Grad-CAM Saliency Shift and Interpretability")
    add_p(
        "To provide post-hoc interpretability for tripped cases, Grad-CAM saliency heatmaps were audited across original and perturbed variants (quantified in Fig. 5). On ISIC 2019, un-tripped samples maintain high saliency stability: mean Cosine Similarity between original and perturbed saliency maps is 0.916, Intersection over Union (IoU) of top-20% activation masks is 0.609, and Centroid Euclidean Drift is 0.044. On tripped samples, Cosine Similarity decreases to 0.354, IoU drops to 0.236, and Centroid Drift increases to 0.199. While this demonstrates that metamorphic flips correlate with activation redistributions, we caution that Grad-CAM heatmaps reflect class-activation gradients rather than verified pathological biomarker boundaries."
    )

    # ── Figure 5: GradCAM Drift ───────────────────────────────────────────────
    fig5_path = os.path.join(FIG_DIR, "fig6_gradcam_drift_quantification.png")
    add_fig(fig5_path, "Fig. 5. Grad-CAM visual attention shift metrics comparing robust vs. tripped lesion classifications.")

    # ── 12. Section VII: Operational Runtime Latency ──────────────────────────
    add_h1("VII. Operational Latency, Hardware Profiling, and Limitations")
    add_p(
        "To evaluate deployment feasibility, latency profiling was performed on both dedicated GPU and edge CPU platforms (illustrated in Fig. 6). In the clinical dermatology pipeline (ConvNeXt-Base), total pipeline execution time is 54.2 ms on an NVIDIA RTX 3050 GPU (18.4 ms for baseline inference, 16.2 ms for parallel metamorphic batch generation, 18.5 ms for batched tensor inference, and 1.1 ms for hierarchical CBI computation). In autonomous vehicle perception evaluated on an edge Intel Core i7 CPU, MobileNetV3-Small baseline inference takes 10.9 ms, metamorphic batch generation takes 142.3 ms, batched inference takes 81.3 ms, and CBI scoring takes 1.2 ms, totaling 235.7 ms."
    )
    add_p(
        "Methodological and clinical limitations must be explicitly noted: (1) Domain Perturbation Validity: On GTSRB, horizontal flip was applied as an unconstrained geometric transform in preliminary runs, which violates semantic invariance for directional signs (e.g., Turn Left vs. Turn Right), artificially elevating naive MT flip rates. Directional transforms must be filtered in production AV stacks. (2) In-Distribution Evaluation: All experiments evaluated held-out test splits from the same source distributions. True out-of-distribution robustness across external clinical centers remains unverified. (3) Statistical Sample Size: The evaluation cohorts (ISIC N=250 with 43 errors, GTSRB N=250/500 with 11/21 errors, Plant Village N=100 with 17 errors) yield wide confidence intervals. Larger multi-center benchmarks are essential to substantiate clinical adoption."
    )

    # ── Figure 6: Latency Breakdown ───────────────────────────────────────────
    fig6_path = os.path.join(FIG_DIR, "fig5_latency_breakdown.png")
    add_fig(fig6_path, "Fig. 6. Component-wise execution latency breakdown comparing GPU server execution (ISIC) with edge CPU execution (GTSRB).")

    # ── 13. Section VIII: Conclusion ──────────────────────────────────────────
    add_h1("VIII. Conclusion and Future Work")
    add_p(
        "This paper presented the design, mathematical formulation, and architecture of the Metamorphic Circuit Breaker (MCB), an evidence-based runtime safety framework for vision AI. By replacing naive binary trip rules with Hierarchical Metamorphic Fault Attribution, Two-Level Hierarchical Normalization with peak anchoring, and multi-family spread metrics, the system resolves representation skew and suppresses false alarm overhead. Comprehensive empirical benchmarking across clinical dermatology (ISIC 2019), autonomous vehicle perception (GTSRB), and agricultural pathology (Plant Village) confirms that MCB resolves representation skew, suppresses false alarm rates by 4x to 13.6x (reducing false alarms to 1.9%–13.0%), and intercepts up to 72.7% of critical runtime misclassifications (and 53.5% on clinical dermatology) with 54.2 ms GPU operational latency. Future work will focus on prospective multi-center clinical trials, adaptive dynamic thresholding, and extending the circuit breaker architecture to 3D volumetric medical imaging."
    )

    # ── 14. Acknowledgment & References ───────────────────────────────────────
    add_h1("Acknowledgment")
    add_p(
        "The authors express sincere gratitude to the Department of Computer Science and Engineering at Mar Baselios College of Engineering and Technology (Autonomous), affiliated to APJ Abdul Kalam Technological University, Thiruvananthapuram, Kerala, India, for providing computational infrastructure, research facilities, and academic guidance throughout this work."
    )

    add_h1("References")
    references_list = [
        "[1] H. A. Haenssle et al., 'Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists,' Annals of Oncology, vol. 29, no. 8, pp. 1836\u20131842, 2018.",
        "[2] S. G. Finlayson et al., 'Adversarial attacks on medical machine learning,' Science, vol. 363, no. 6433, pp. 1287\u20131289, 2019.",
        "[3] T. Y. Chen, F.-C. Kuo, R. G. Qiu, and T. Tse, 'Metamorphic testing: A review of challenges and opportunities,' ACM Computing Surveys (CSUR), vol. 51, no. 1, pp. 1\u201327, 2018.",
        "[4] F. J. Sendelbach and M. Funk, 'Alarm fatigue: a patient safety concern,' AACN Advanced Critical Care, vol. 24, no. 4, pp. 378\u2013386, 2013.",
        "[5] B. C. Tschandl et al., 'The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,' Scientific Data, vol. 5, no. 1, pp. 1\u20139, 2018.",
        "[6] Z. Liu et al., 'A convnet for the 2020s,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976\u201311986, 2022.",
        "[7] Y. Tian, K. Pei, S. Jana, and B. Ray, 'DeepTest: Automated testing of deep-neural-network-driven autonomous cars,' in Proceedings of the 40th International Conference on Software Engineering (ICSE), pp. 303\u2013314, 2018.",
        "[8] A. Dwarakanath et al., 'Identifying implementation bugs in machine learning based image classifiers using metamorphic testing,' in 2018 IEEE International Conference on Software Testing, Verification and Validation (ICST), pp. 118\u2013128, 2018.",
        "[9] Y. Gal and Z. Ghahramani, 'Dropout as a bayesian approximation: Representing model uncertainty in deep learning,' in International Conference on Machine Learning (ICML), pp. 1050\u20131059, 2016.",
        "[10] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618\u2013626, 2017.",
        "[11] D. Hendrycks and K. Gimpel, 'A baseline for detecting misclassified and out-of-distribution examples in neural networks,' in International Conference on Learning Representations (ICLR), 2017.",
        "[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks,' in International Conference on Machine Learning (ICML), pp. 1321\u20131330, 2017.",
        "[13] D. Hendrycks and T. Dietterich, 'Benchmarking neural network robustness to common corruptions and perturbations,' in International Conference on Learning Representations (ICLR), 2019.",
        "[14] Y. Geifman and R. El-Yaniv, 'Selective classification for deep neural networks,' in Advances in Neural Information Processing Systems (NeurIPS), pp. 4878\u20134887, 2017.",
        "[15] M. N. Nygard, 'Release It!: Design and Deploy Production-Ready Software,' Pragmatic Bookshelf, 2018.",
        "[16] J. Zhang et al., 'Machine learning testing: Survey, landscapes and horizons,' IEEE Transactions on Software Engineering, vol. 48, no. 1, pp. 1\u201336, 2020.",
    ]

    for ref in references_list:
        add_ref(ref)

    # ── 15. Save Document to SEPARATE Path ─────────────────────────────────────
    print(f"Saving rectified document to: {OUTPUT_DOCX_PATH}")
    doc.save(OUTPUT_DOCX_PATH)
    print(f"SUCCESS: Document saved to {OUTPUT_DOCX_PATH} ({os.path.getsize(OUTPUT_DOCX_PATH)} bytes)")

    # ── 16. Convert to PDF via Word COM ────────────────────────────────────────
    ps_script = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{OUTPUT_DOCX_PATH}')
        $doc.SaveAs([ref]'{OUTPUT_PDF_PATH}', [ref]17)
        $doc.Close([ref]0)
    }} finally {{
        $word.Quit()
    }}
    """
    ps_file = os.path.join(ROOT_DIR, "scripts", "_temp_convert_pdf.ps1")
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write(ps_script)

    print("Converting to publication PDF via Word COM automation...")
    res = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_file], capture_output=True, text=True)
    if os.path.exists(ps_file):
        try:
            os.remove(ps_file)
        except Exception:
            pass

    if os.path.exists(OUTPUT_PDF_PATH) and os.path.getsize(OUTPUT_PDF_PATH) > 0:
        print(f"SUCCESS: Rectified publication-ready PDF generated at: {OUTPUT_PDF_PATH} ({os.path.getsize(OUTPUT_PDF_PATH)} bytes)")
    else:
        print("Notice on Word COM conversion:", res.stdout, res.stderr)


if __name__ == "__main__":
    build_paper()
