"""
scripts/update_and_build_ieee_paper.py — Update the IEEE Research Paper with
verified empirical results, full Table I, figures, and statistical justifications across active domains.
Converts both to .docx and publication-ready .pdf via Word COM automation.
"""

import os
import io
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
import latex2mathml.converter
import lxml.etree as ET
import subprocess

ROOT_DIR = r"C:\Users\Goutham C S\Desktop\main_implementation"
TEMPLATE_PATH = os.path.join(ROOT_DIR, "Research Paper", "conference-template-a4.docx")
OUTPUT_DOCX_PATH = os.path.join(ROOT_DIR, "Research Paper", "Metamorphic_Circuit_Breaker_IEEE_Paper_Latest.docx")
PRIMARY_DOCX_PATH = os.path.join(ROOT_DIR, "Research Paper", "Metamorphic_Circuit_Breaker_IEEE_Paper.docx")
PRIMARY_PDF_PATH = os.path.join(ROOT_DIR, "Research Paper", "Metamorphic_Circuit_Breaker_IEEE_Paper.pdf")
FIG_DIR = os.path.join(ROOT_DIR, "outputs", "evaluation_report", "figures")
XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"

xslt_doc = ET.parse(XSL_PATH)
transform_mml = ET.XSLT(xslt_doc)


def latex_to_omml(latex_code: str, bold: bool = True):
    mathml = latex2mathml.converter.convert(latex_code)
    mathml_doc = ET.fromstring(mathml)
    omml_doc = transform_mml(mathml_doc)
    ns = {
        'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    }

    for t in omml_doc.xpath('.//m:t', namespaces=ns):
        if t.text:
            t.text = t.text.replace('\xa0', ' ')

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

    return ET.tostring(omml_doc, encoding="unicode")


def set_ieee_table_borders(table):
    tblPr = table._tbl.tblPr
    existing_borders = tblPr.find(qn('w:tblBorders'))
    if existing_borders is not None:
        tblPr.remove(existing_borders)

    borders_xml = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders_xml)

    header_row = table.rows[0]._tr.get_or_add_trPr()
    hdr_border = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>\n'
        f'</w:tcBorders>'
    )
    for cell in table.rows[0].cells:
        cell._tc.get_or_add_tcPr().append(parse_xml(
            f'<w:tcBorders {nsdecls("w")}>\n'
            f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
            f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>\n'
            f'  <w:left w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'</w:tcBorders>'
        ))


def main():
    print("Loading IEEE conference template...")
    doc = docx.Document(TEMPLATE_PATH)

    for p in list(doc.paragraphs):
        p._p.getparent().remove(p._p)
    for t in list(doc.tables):
        t._tbl.getparent().remove(t._tbl)

    def add_p(text, bold_prefix=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p = doc.add_paragraph(style=doc.styles["Standard"])
        p.alignment = align
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.05
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.bold = True
            r_b.font.name = "Times New Roman"
            r_b.font.size = Pt(9.5)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        return p

    def add_h1(text):
        p = doc.add_paragraph(style=doc.styles["Heading 1"])
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        return p

    def add_h2(text):
        p = doc.add_paragraph(style=doc.styles["Heading 2"])
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.italic = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        return p

    def add_equation_omml(latex_code: str, eq_num: str):
        p = doc.add_paragraph(style=doc.styles["Standard"])
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(4)
        omml_xml = latex_to_omml(latex_code, bold=True)
        oMathPara = parse_xml(
            f'<m:oMathPara {nsdecls("m", "w")}>\n'
            f'  <m:oMathParaPr>\n'
            f'    <m:jc m:val="center"/>\n'
            f'  </m:oMathParaPr>\n'
            f'  {omml_xml}\n'
            f'</m:oMathPara>'
        )
        p._p.append(oMathPara)
        p_num = doc.add_paragraph(style=doc.styles["Standard"])
        p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_num.paragraph_format.space_before = Pt(0)
        p_num.paragraph_format.space_after = Pt(4)
        r_num = p_num.add_run(f"({eq_num})")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(9)
        return p

    def add_fig(img_path, caption, width=Inches(3.3)):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph(style=doc.styles["Standard"])
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            run = p_img.add_run()
            run.add_picture(img_path, width=width)
            p_cap = doc.add_paragraph(style=doc.styles["figure caption"])
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(6)
            r_cap = p_cap.add_run(caption)
            r_cap.italic = True
            r_cap.font.name = "Times New Roman"
            r_cap.font.size = Pt(8)

    def add_ref(text):
        p = doc.add_paragraph(style=doc.styles["Standard"])
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        return p

    # ── Title ─────────────────────────────────────────────────────────────────
    p_title = doc.add_paragraph(style=doc.styles["title"])
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(12)
    r_t = p_title.add_run(
        "Model-Agnostic Metamorphic Circuit Breaker: Mitigating Runtime Fragility in Safety-Critical Computer Vision"
    )
    r_t.bold = True
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(22)

    # ── Authors ───────────────────────────────────────────────────────────────
    authors_data = [
        {"name": "Goutham C S", "email": "gouthamcs.cs2125@mbcet.ac.in"},
        {"name": "Joel Varghese", "email": "joelvarghese.cs2125@mbcet.ac.in"},
        {"name": "Joshua T S", "email": "joshuats.cs2125@mbcet.ac.in"},
        {"name": "Kevin Jacob", "email": "kevinjacob.cs2125@mbcet.ac.in"},
        {"name": "Dr. Jesna Mohan", "email": "jesna.mohan@mbcet.ac.in", "role": "Associate Professor"},
    ]

    p_a1 = doc.add_paragraph(style=doc.styles["Standard"])
    p_a1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_a1.paragraph_format.space_before = Pt(0)
    p_a1.paragraph_format.space_after = Pt(2)
    for i in range(3):
        a = authors_data[i]
        r = p_a1.add_run(f"{a['name']}\nDept. of Computer Science & Engineering\nMar Baselios College of Engineering and Technology\n{a['email']}" + ("\t\t" if i < 2 else ""))
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)

    p_a2 = doc.add_paragraph(style=doc.styles["Standard"])
    p_a2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_a2.paragraph_format.space_before = Pt(4)
    p_a2.paragraph_format.space_after = Pt(10)
    for i in range(3, 5):
        a = authors_data[i]
        role_txt = f"\n{a['role']}" if "role" in a else ""
        r = p_a2.add_run(f"{a['name']}{role_txt}\nDept. of Computer Science & Engineering\nMar Baselios College of Engineering and Technology\n{a['email']}" + ("\t\t\t" if i == 3 else ""))
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)

    # ── Abstract ──────────────────────────────────────────────────────────────
    p_abs = doc.add_paragraph(style=doc.styles["Abstract"])
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(6)
    p_abs.paragraph_format.space_after = Pt(4)
    r_ab = p_abs.add_run("Abstract—")
    r_ab.bold = True
    r_ab.italic = True
    r_ab.font.name = "Times New Roman"
    r_ab.font.size = Pt(9)
    r_txt = p_abs.add_run(
        "Deep convolutional and transformer networks deployed in safety-critical computer vision domains—such as dermatological oncology, autonomous vehicle perception, and precision agriculture—exhibit severe transform-brittleness: producing confident, correct classifications on canonical inputs that catastrophically flip under subtle, label-preserving perturbations. Traditional runtime monitoring techniques either exacerbate operator alert fatigue through uncalibrated binary checks or incur prohibitive computational latency. This paper presents the Metamorphic Circuit Breaker (MCB), an evidence-based runtime safety layer grounded in Hierarchical Metamorphic Fault Attribution. Candidate input transformations are structured into orthogonal semantic families (Geometric, Photometric, Sensor/Noise), each stratified into non-linear severity ladders. To resolve transformation-type imbalance skew, we introduce Two-Level Hierarchical Normalization, ensuring continuous multi-tier transforms cannot overpower single-tier tests. Runtime safety transitions across three operational states (CLOSED, HALF-OPEN, OPEN) are governed by the Composite Brittleness Index (CBI), combining peak family instability, cross-family spread, and model doubt. Extensive cross-domain evaluations across ConvNeXt-Base on ISIC 2019 dermoscopy, MobileNetV3-Small on GTSRB traffic signs, and EfficientNet-B0 on Plant Village crop pathology demonstrate that MCB suppresses naive metamorphic false alarms by 4x to 13.6x (reducing false trip rates from 84.1% to 1.9%-20.9%), while capturing silent catastrophic failures missed by Softmax Entropy. Coupled with Gradient-weighted Class Activation Mapping (GradCAM) explainability, the framework provides human operators with causal visual audits of spatial attention drift upon safety interception."
    )
    r_txt.font.name = "Times New Roman"
    r_txt.font.size = Pt(9)

    p_idx = doc.add_paragraph(style=doc.styles["Index Terms"])
    p_idx.paragraph_format.space_before = Pt(2)
    p_idx.paragraph_format.space_after = Pt(8)
    r_ib = p_idx.add_run("Index Terms—")
    r_ib.bold = True
    r_ib.italic = True
    r_ib.font.name = "Times New Roman"
    r_ib.font.size = Pt(9)
    r_it = p_idx.add_run("AI Safety, Metamorphic Testing, Runtime Fault Attribution, Circuit Breaker Pattern, Dermoscopic Oncology, Autonomous Driving, Explainable AI.")
    r_it.font.name = "Times New Roman"
    r_it.font.size = Pt(9)

    # ── Section I: Introduction ───────────────────────────────────────────────
    add_h1("I. Introduction")
    add_p(
        "Deep learning vision models achieve dermatological diagnostic parity with board-certified physicians [1] and enable real-time perceptual parsing in autonomous transit systems. Despite high nominal test accuracy, deep neural networks exhibit catastrophic brittleness: small, label-preserving perturbations (such as sensor defocus, illumination shifts, or minute tilts) can invert high-confidence correct inferences into erroneous outputs [2]."
    )
    add_p(
        "In safety-critical workflows, silent misclassifications carry catastrophic liability: failing to detect invasive melanoma delays surgical excision, and misidentifying speed limit signs in autonomous driving induces severe vehicle control faults. While Metamorphic Testing (MT) [3] addresses the test oracle problem by evaluating consistency across invariant input relations, deploying naive binary MT checks directly into production systems creates a crippling dilemma: flagging a fault upon any single prediction shift induces overwhelming false alarm rates (exceeding 80%), creating operator alert fatigue [4] that leads human supervisors to disable the safety shield entirely."
    )
    add_p(
        "To resolve these challenges, this paper introduces the Metamorphic Circuit Breaker (MCB), a runtime safety framework founded on Hierarchical Metamorphic Fault Attribution. Our core contributions are:"
    )
    add_p("• Hierarchical Semantic Grouping & Severity Ladders: Organizing metamorphic relations into orthogonal families (Geometric, Photometric, Sensor/Noise), each stratified across mild, moderate, and severe parameter tiers with non-linear penalty weighting.")
    add_p("• Skew-Free Two-Level Normalization: Mathematically resolving transformation-type imbalance skew, ensuring continuous multi-step transforms do not overpower single-tier tests.")
    add_p("• Evidence-Based Three-State Circuit Breaker: Formulating the Composite Brittleness Index (CBI) to govern transitions across CLOSED (healthy auto-approval), HALF-OPEN (provisional output with caution telemetry), and OPEN (fail-safe interception to doctor triage).")
    add_p("• Cross-Domain Verification & Saliency Explainability: Validating empirical efficacy across multiple architectures (ConvNeXt-Base, MobileNetV3-Small, EfficientNet-B0) and pairing safety trips with GradCAM spatial attention drift diagnostics.")

    # ── Section II: Related Work ──────────────────────────────────────────────
    add_h1("II. Related Work")
    add_h2("A. Deep Learning in Safety-Critical Vision")
    add_p(
        "Automated vision pipelines have shifted from hand-crafted feature extractors to modern deep backbones like ConvNeXt [6], which combines depthwise 7x7 convolutions with inverted bottlenecks to achieve competitive accuracy while preserving computational throughput. Nevertheless, convolutional and transformer inductive biases remain fragile to out-of-distribution high-frequency perturbations."
    )
    add_h2("B. Metamorphic Testing in Vision Systems")
    add_p(
        "Pioneered by Chen et al. [3], metamorphic testing verifies relational properties across multiple program executions. While MT has audited autonomous driving simulators [7] and deep learning bugs [8], prior literature focuses overwhelmingly on offline testing. Operating MT as an active, runtime safety circuit breaker requires rigorous false-alarm suppression and deterministic latency bounding."
    )
    add_h2("C. Uncertainty Estimation and Out-of-Distribution Gating")
    add_p(
        "Runtime safety has traditionally relied on Monte Carlo Dropout (MCDO) [9] and Softmax Entropy thresholding. However, deep networks frequently suffer from severe softmax overconfidence, assigning >95% probability to catastrophic errors. Furthermore, stochastic sampling incurs a 10x compute penalty. The proposed MCB framework circumvents internal parameter sampling by testing behavioral invariance across calibrated input manifolds."
    )

    # ── Section III: Architectural Lineage (Phase 1 to Phase 2) ───────────────
    add_h1("III. Architectural Foundation: Bridging Phase 1 and Phase 2")
    add_p(
        "This research establishes a direct engineering bridge from the foundational Phase 1 implementation (operational baseline) to the Phase 2 Hierarchical Fault Attribution framework:"
    )
    add_p("1) Phase 1 Baseline: Ingestion of ISIC 2019 (25,331 clinical dermatoscope images across 8 classes) fine-tuned on ConvNeXt-Base, evaluated against 8 empirical metamorphic relations, and paired with PyTorch GradCAM hooks and a Streamlit research dashboard.")
    add_p("2) Phase 2 Innovation: Introduction of 22 multi-tier relations across 3 orthogonal families, Two-Level Hierarchical Normalization, Cross-Family Spread (CFS) scoring, Composite Brittleness Index (CBI), and a real-time FastAPI/React clinical intervention portal.")

    # ── Section IV: Mathematical Formulation ──────────────────────────────────
    add_h1("IV. Mathematical Formulation and Decision Logic")
    add_h2("A. Severity Weighting Function")
    add_p("A prediction flip under an imperceptible mild perturbation signals severe model fragility, whereas a flip under aggressive boundary degradation is penalized less. We formalize this via a decreasing severity weighting function w(s):")
    add_equation_omml(
        r"w(s) = \begin{cases} 1.0, & s = \text{Mild} \\ 0.6, & s = \text{Moderate} \\ 0.3, & s = \text{Severe} \end{cases}",
        "1"
    )

    add_h2("B. Resolving Representation Skew via Two-Level Normalization")
    add_p("In flat metamorphic scoring, transformation types with more tiers artificially overpower single-tier tests. We resolve this via Two-Level Hierarchical Normalization:")
    add_p("Level 1 (Type Score): For each transformation type m in family k, compute the normalized severity score:")
    add_equation_omml(
        r"\text{Score}_m(x) = \frac{\sum_{s \in \mathcal{S}_m} w(s) \cdot \mathbb{I}[f(t_{m,s}(x)) \ne y_0]}{\sum_{s \in \mathcal{S}_m} w(s)}",
        "2"
    )
    add_p("Level 2 (Intra-Family Instability Ratio): Unweighted arithmetic mean across all M_k transformation types in family k:")
    add_equation_omml(
        r"\text{IR}_k(x) = \frac{1}{M_k} \sum_{m=1}^{M_k} \text{Score}_m(x)",
        "3"
    )
    add_p("This formulation guarantees that every transformation type receives an equal 1 / M_k share of the family evaluation, eliminating representation skew.")

    add_h2("C. Cross-Family Spread Score and the 35% Activation Boundary")
    add_p("Simultaneous failure across independent families signals systemic reasoning collapse:")
    add_equation_omml(
        r"\text{CFS}(x) = \frac{1}{|\mathcal{F}|} \sum_{k=1}^{|\mathcal{F}|} \mathbb{I}[\text{IR}_k(x) \ge \tau_{\text{fam}}]",
        "4"
    )
    add_p("where |F| = 3 and tau_fam is the family compromise cutoff (default 0.35).")

    add_h2("D. Composite Brittleness Index (CBI) and Three-State Decision Logic")
    add_equation_omml(
        r"\text{CBI}(x) = \alpha \cdot \max_{k} \text{IR}_k(x) + \beta \cdot \text{CFS}(x) + \gamma \cdot (1 - c_0)",
        "5"
    )
    add_p("anchored with default parameters alpha = 0.50, beta = 0.35, and gamma = 0.15.")
    add_equation_omml(
        r"\text{State}(x) = \begin{cases} \text{CLOSED}, & \text{CBI}(x) < \theta_{\text{warn}} \ (0.25) \\ \text{HALF-OPEN}, & \theta_{\text{warn}} \le \text{CBI}(x) < \theta_{\text{trip}} \ (0.55) \\ \text{OPEN}, & \text{CBI}(x) \ge \theta_{\text{trip}} \ (0.55) \end{cases}",
        "6"
    )

    # ── Section V: Explainability ─────────────────────────────────────────────
    add_h1("V. Clinical Human-in-the-Loop Triage and Saliency Explainability")
    add_p(
        "When an incoming inference trips the circuit breaker to OPEN, the automated result is halted, and native PyTorch hooks on the penultimate feature extraction layer compute Gradient-weighted Class Activation Mapping (GradCAM) [10]. By comparing baseline saliency against perturbed saliency, the system provides reviewing clinicians with visual evidence explaining whether model attention drifted from diagnostic lesions onto healthy skin or sensor artifacts."
    )

    # ── Section VI: Empirical Results ─────────────────────────────────────────
    add_h1("VI. Empirical Evaluation and Experimental Results")
    add_p(
        "The proposed MCB framework was subjected to rigorous empirical evaluation across three active vision domains: clinical dermatology (ConvNeXt-Base on ISIC 2019, N=250 held-out images), autonomous traffic sign perception (MobileNetV3-Small on GTSRB, N=500 images), and agricultural crop pathology (EfficientNet-B0 on Plant Village, N=100 held-out images). All tests conform to the standardized 7-Test evaluation suite."
    )

    add_h2("A. Competing Baseline Safety Paradigm Comparison")
    add_p(
        "Table I compiles the formal benchmark results comparing the proposed Hierarchical MCB against prevalent runtime safety mechanisms on identical held-out test splits."
    )

    # Table I
    p_tbl_head = doc.add_paragraph(style=doc.styles["table head"])
    p_tbl_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tbl_head.paragraph_format.space_before = Pt(8)
    p_tbl_head.paragraph_format.space_after = Pt(3)
    r_th = p_tbl_head.add_run("TABLE I\nCOMPETING RUNTIME SAFETY PARADIGM COMPARISON ACROSS HETEROGENEOUS DOMAINS")
    r_th.font.name = "Times New Roman"
    r_th.font.size = Pt(8.5)

    headers = ["Domain & Backbone", "Safety Mechanism", "Safety Recall (%)", "False Trip Rate (%)", "Precision (%)", "F2-Score", "P95 Latency (ms)"]
    rows = [
        ["Dermatology\nConvNeXt-Base\n(ISIC 2019, N=250)", "Standalone (Unmonitored)\nNaive Binary 1-Flip MT\nSoftmax Entropy Gating\nMonte Carlo Dropout\nProposed MCB (Default)\nProposed MCB (Calibrated)", "0.0%\n60.5%\n41.9%\n34.9%\n16.3%\n53.5%", "0.0%\n20.3%\n9.7%\n11.1%\n1.9%\n13.0%", "n/a\n38.2%\n47.4%\n39.5%\n63.6%\n46.0%", "0.000\n0.542\n0.429\n0.357\n0.191\n0.518", "143.2 ms\n2537.4 ms\n143.2 ms\n1432.0 ms\n2537.4 ms\n2537.4 ms"],
        ["Autonomous Driving\nMobileNetV3-Small\n(GTSRB Signs, N=250/500)", "Standalone (Unmonitored)\nNaive Binary 1-Flip MT\nSoftmax Entropy Gating\nMonte Carlo Dropout\nProposed MCB (Default)\nProposed MCB (Calibrated)", "0.0%\n100.0%\n63.6%\n90.9%\n72.7%\n90.5%", "0.0%\n84.1%\n18.0%\n16.7%\n20.9%\n24.4%", "n/a\n12.8%\n15.2%\n21.3%\n15.2%\n14.0%", "0.000\n0.215\n0.372\n0.532\n0.392\n0.421", "13.8 ms\n295.6 ms\n13.8 ms\n131.1 ms\n295.6 ms\n295.6 ms"],
        ["Crop Pathology\nEfficientNet-B0\n(Plant Village, N=100)", "Standalone (Unmonitored)\nNaive Binary 1-Flip MT\nSoftmax Entropy Gating\nMonte Carlo Dropout\nProposed MCB (Default)\nProposed MCB (Calibrated)", "0.0%\n70.6%\n29.4%\n17.6%\n17.6%\n35.3%", "0.0%\n81.9%\n6.0%\n8.4%\n6.0%\n6.0%", "0.0%\n15.0%\n50.0%\n30.0%\n37.5%\n54.5%", "0.000\n0.405\n0.321\n0.192\n0.197\n0.380", "1302.9 ms\n7927.2 ms\n1302.9 ms\n13029.0 ms\n7927.2 ms\n7927.2 ms"],
    ]

    tbl_bench = doc.add_table(rows=len(rows)+1, cols=len(headers))
    tbl_bench.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col_idx, h in enumerate(headers):
        cell = tbl_bench.cell(0, col_idx)
        cell.text = h
        p_c = cell.paragraphs[0]
        p_c.runs[0].bold = True
        p_c.runs[0].font.name = "Times New Roman"
        p_c.runs[0].font.size = Pt(7.5)
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row_idx, r_vals in enumerate(rows):
        for col_idx, val in enumerate(r_vals):
            cell = tbl_bench.cell(row_idx + 1, col_idx)
            cell.text = val
            p_c = cell.paragraphs[0]
            p_c.runs[0].font.name = "Times New Roman"
            p_c.runs[0].font.size = Pt(7.0)
            if col_idx > 1:
                p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_ieee_table_borders(tbl_bench)

    p_tbl_foot = doc.add_paragraph(style=doc.styles["table footnote"])
    p_tbl_foot.paragraph_format.space_before = Pt(2)
    p_tbl_foot.paragraph_format.space_after = Pt(6)
    r_tf = p_tbl_foot.add_run("*Note: Evaluated across held-out test splits. Default thresholds set to (tau=0.35, theta=0.55); calibrated thresholds derived via Test 5 Youden J optimization.")
    r_tf.font.name = "Times New Roman"
    r_tf.font.size = Pt(7.0)

    # Insert Figure 2: Baseline Paradigm Comparison
    add_fig(os.path.join(FIG_DIR, "fig2_baseline_paradigm_comparison.png"), "Fig. 1. Safety Recall vs. False Alarm Rate across evaluated runtime safety paradigms across all active domains.")

    add_h2("B. Suppression of Naive Metamorphic Alert Fatigue")
    add_p(
        "A critical empirical discovery from Table I and Fig. 1 is the profound operational failure of naive binary metamorphic testing in production settings. While Naive 1-Flip MT achieves high raw recall (60.5% on ISIC, 100% on GTSRB, 70.6% on Plant Village), it triggers catastrophic false alarm rates (20.3% on ISIC, 84.1% on GTSRB, 81.9% on Plant Village). In an autonomous driving vehicle, tripping safety alarms on 84.1% of benign, correctly classified signs induces extreme driver fatigue, rendering the system practically unusable. In contrast, the proposed Hierarchical MCB reduces false alarms by 4x to 13.6x (achieving 1.9% on ISIC, 20.9% on GTSRB, and 6.0% on Plant Village at default settings)."
    )

    add_h2("C. Transformation Family Profiling and Two-Level Skew Elimination")
    add_p(
        "Test 4 profiles the semantic vulnerability of vision models. Across all three domains, Sensor / Noise emerged as the undisputed primary failure axis: mean IR = 0.0475 on ISIC, 0.2422 on GTSRB, and 0.1792 on Plant Village (Fig. 2). Furthermore, our ablation study empirically proved the mathematical necessity of Two-Level Hierarchical Normalization: on ConvNeXt ISIC, Two-Level scoring achieved an AUROC of 0.813 versus 0.728 for flat averaging, successfully intercepting 9 critical diagnostic errors that flat averaging completely drowned out."
    )

    # Insert Figure 1: Radar Chart
    add_fig(os.path.join(FIG_DIR, "fig1_domain_radar.png"), "Fig. 2. Cross-domain transformation family instability ratio (IR_k) profile, confirming Sensor/Noise as the universal primary failure trigger.")

    add_h2("D. Operating Threshold Calibration & Method A Weight Ablation")
    add_p(
        "Test 5 evaluated 119 cutoff pairs (tau_fam in [0.20, 0.50] x theta_trip in [0.10, 0.90]). At documented theoretical defaults (0.35 / 0.55), the MCB acts as an ultra-conservative, near-zero-false-alarm filter (1.9% FPR on ISIC, 6.0% FPR on Plant Village). Data-driven calibration via Youden's Index J = Recall - FPR identifies domain-specific operating sweet spots: (tau=0.20, theta=0.10, J=0.404) for ConvNeXt, (tau=0.25, theta=0.30, J=0.661) for MobileNetV3, and (tau=0.45, theta=0.50, J=0.293) for EfficientNet (Fig. 3). Under calibrated cutoffs, safety recall rises to 53.5% on ISIC and 90.5% on GTSRB."
    )
    add_p(
        "To evaluate formula weights without threshold bias, Method A evaluated cutoff-free Area Under the ROC Curve (AUROC) across 5 configurations. High Confidence Trust (Config 4: alpha=0.25, beta=0.25, gamma=0.50) achieved the highest AUROC (0.823 on ISIC, 0.907 on GTSRB, 0.828 on Plant Village), with the Proposed Balanced Formula (Config 1) closely following (0.813 on ISIC, 0.885 on GTSRB). Crucially, the Pure Black-Box Formula (Config 5: gamma=0.00) preserved strong discrimination (0.721 on ISIC, 0.869 on GTSRB), proving that the framework operates effectively even when model confidence is inaccessible."
    )

    # Insert Figure 3 & Figure 4
    add_fig(os.path.join(FIG_DIR, "fig3_threshold_calibration_shifts.png"), "Fig. 3. Calibrated operating cutoffs (tau_fam, theta_trip) and peak Youden Index J across evaluated architectures.")
    add_fig(os.path.join(FIG_DIR, "fig4_method_a_weight_ablation.png"), "Fig. 4. Method A cutoff-free AUROC score across 5 formula weight configurations, validating multi-family scoring stability.")

    add_h2("E. Qualitative Explainability and Grad-CAM Attention Shift Audit")
    add_p(
        "Test 7 established causal verification of safety trips using GradCAM attention maps. On stable inputs, visual attention remains tightly localized to diagnostic lesion structures (Mean Cosine = 0.916, Top-20% IoU = 0.609, Centroid Drift = 0.044). In contrast, on tripped inputs, visual saliency radically collapses onto surrounding normal skin or illumination artifacts (Mean Cosine = 0.354, IoU = 0.236, Centroid Drift surges 4.5x to 0.199), as quantified in Fig. 5. This provides human clinicians with definitive visual proof of feature extraction breakdown."
    )

    # Insert Figure 6: GradCAM Audit
    add_fig(os.path.join(FIG_DIR, "fig6_gradcam_drift_quantification.png"), "Fig. 5. Quantitative GradCAM attention drift audit comparing stable approved inputs against tripped safety interventions on ConvNeXt-Base ISIC.")

    # ── Section VII: Cross-Domain Generalization ──────────────────────────────
    add_h1("VII. Cross-Domain Generalization & Deployment Feasibility")
    add_p(
        "The cross-domain results across autonomous driving (MobileNetV3 on GTSRB) and crop pathology (EfficientNet on Plant Village) prove that the MCB architecture is strictly model-agnostic and domain-adaptive. Furthermore, latency profiling in Test 6 confirmed operational feasibility: batched forward-pass execution executes in 81.3 ms on lightweight mobile backbones, satisfying real-time bounded monitoring requirements, while Python scoring logic executes in under 1.0 ms across all platforms."
    )

    # ── Section VIII: Conclusion ──────────────────────────────────────────────
    add_h1("VIII. Conclusion")
    add_p(
        "This paper presented the empirical evaluation and mathematical validation of the Metamorphic Circuit Breaker (MCB), a model-agnostic runtime AI safety layer. By replacing naive flat metamorphic checks with Two-Level Hierarchical Normalization and Composite Brittleness Index gating, the system suppresses operator alert fatigue by up to 13.6x while intercepting critical silent failures that bypass Softmax Entropy. Through multi-domain empirical testing across oncology, vehicular perception, and crop pathology, the framework proves both its generalizability and real-time execution viability, providing an actionable runtime integrity shield for high-stakes computer vision."
    )

    # ── Acknowledgment & References ───────────────────────────────────────────
    add_h1("Acknowledgment")
    add_p(
        "The authors express sincere gratitude to the Department of Computer Science and Engineering at Mar Baselios College of Engineering and Technology (Autonomous), affiliated to APJ Abdul Kalam Technological University, Thiruvananthapuram, Kerala, India, for research facilities and academic guidance."
    )

    add_h1("References")
    references_list = [
        "[1] H. A. Haenssle et al., 'Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists,' Annals of Oncology, vol. 29, no. 8, pp. 1836–1842, 2018.",
        "[2] S. G. Finlayson et al., 'Adversarial attacks on medical machine learning,' Science, vol. 363, no. 6433, pp. 1287–1289, 2019.",
        "[3] T. Y. Chen, F.-C. Kuo, R. G. Qiu, and T. Tse, 'Metamorphic testing: A review of challenges and opportunities,' ACM Computing Surveys (CSUR), vol. 51, no. 1, pp. 1–27, 2018.",
        "[4] F. J. Sendelbach and M. Funk, 'Alarm fatigue: a patient safety concern,' AACN Advanced Critical Care, vol. 24, no. 4, pp. 378–386, 2013.",
        "[5] N. Codella et al., 'Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC),' arXiv preprint arXiv:1902.03368, 2019.",
        "[6] Z. Liu et al., 'A convnet for the 2020s,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986, 2022.",
        "[7] Y. Tian, K. Pei, S. Jana, and B. Ray, 'DeepTest: Automated testing of deep-neural-network-driven autonomous cars,' in Proceedings of the 40th International Conference on Software Engineering (ICSE), pp. 303–314, 2018.",
        "[8] A. Dwarakanath et al., 'Identifying implementation bugs in machine learning based image classifiers using metamorphic testing,' in 2018 IEEE International Conference on Software Testing, Verification and Validation (ICST), pp. 118–128, 2018.",
        "[9] Y. Gal and Z. Ghahramani, 'Dropout as a bayesian approximation: Representing model uncertainty in deep learning,' in International Conference on Machine Learning (ICML), pp. 1050–1059, 2016.",
        "[10] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618–626, 2017.",
        "[11] D. Hendrycks and K. Gimpel, 'A baseline for detecting misclassified and out-of-distribution examples in neural networks,' in International Conference on Learning Representations (ICLR), 2017.",
        "[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks,' in International Conference on Machine Learning (ICML), pp. 1321–1330, 2017.",
        "[13] D. Amodei et al., 'Concrete problems in AI safety,' arXiv preprint arXiv:1606.06565, 2016.",
        "[14] M. N. Nygard, 'Release It!: Design and Deploy Production-Ready Software,' Pragmatic Bookshelf, 2018.",
        "[15] J. Zhang et al., 'Machine learning testing: Survey, landscapes and horizons,' IEEE Transactions on Software Engineering, vol. 48, no. 1, pp. 1–36, 2020.",
        "[16] P. Tschandl, C. Rosendahl, and H. Kittler, 'The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,' Scientific Data, vol. 5, no. 1, pp. 1–9, 2018.",
    ]

    for ref in references_list:
        add_ref(ref)

    # ── Save DOCX ─────────────────────────────────────────────────────────────
    target_docx = PRIMARY_DOCX_PATH
    try:
        doc.save(target_docx)
        print(f"SUCCESS: IEEE Research Paper DOCX saved to: {target_docx}")
    except PermissionError:
        target_docx = OUTPUT_DOCX_PATH
        doc.save(target_docx)
        print(f"NOTICE: Primary file locked; saved to: {target_docx}")

    # ── Convert to PDF via Word COM ───────────────────────────────────────────
    print("Converting DOCX to PDF via Word COM automation...")
    ps_cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Open('{target_docx}')
    $doc.SaveAs([ref] '{PRIMARY_PDF_PATH}', [ref] 17)
    $doc.Close()
    $word.Quit()
    """
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    if os.path.exists(PRIMARY_PDF_PATH) and os.path.getsize(PRIMARY_PDF_PATH) > 0:
        print(f"SUCCESS: Publication-ready IEEE Paper PDF generated at: {PRIMARY_PDF_PATH} (size: {os.path.getsize(PRIMARY_PDF_PATH)} bytes)")
    else:
        print("Word COM conversion output:", res.stdout, res.stderr)


if __name__ == "__main__":
    main()
