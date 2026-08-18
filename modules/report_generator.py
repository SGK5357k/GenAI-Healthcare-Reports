from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)

import os
import re


# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf(report_text, filename="synthetic_medical_report.pdf"):
    """
    Convert the generated Markdown-style medical report
    into a professional PDF.

    Parameters
    ----------
    report_text : str
        Generated medical report.

    filename : str
        Output PDF filename.

    Returns
    -------
    str
        Path of generated PDF.
    """

    # --------------------------------------------------------
    # Make sure output directory exists
    # --------------------------------------------------------

    output_dir = "generated_reports"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # If only filename is supplied,
    # put it inside generated_reports/

    if not os.path.dirname(filename):

        filepath = os.path.join(
            output_dir,
            filename
        )

    else:

        filepath = filename

        parent = os.path.dirname(filepath)

        if parent:

            os.makedirs(
                parent,
                exist_ok=True
            )

    # --------------------------------------------------------
    # PDF document
    # --------------------------------------------------------

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MedicalTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "MedicalHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=7
    )

    subheading_style = ParagraphStyle(
        "MedicalSubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        spaceBefore=7,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "MedicalBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "MedicalBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-7,
        bulletIndent=3,
        spaceAfter=5
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8.5,
        leading=12,
        spaceBefore=10
    )

    # --------------------------------------------------------
    # Story
    # --------------------------------------------------------

    story = []

    lines = report_text.splitlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # ----------------------------------------------------
        # Empty line
        # ----------------------------------------------------

        if not line:

            story.append(
                Spacer(1, 4)
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Main title
        # ----------------------------------------------------

        if line.startswith("# ") and not line.startswith("## "):

            title = line[2:].strip()

            story.append(
                Paragraph(
                    escape_html(title),
                    title_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Section heading
        # ----------------------------------------------------

        if line.startswith("## "):

            heading = line[3:].strip()

            story.append(
                Paragraph(
                    escape_html(heading),
                    heading_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Subsection heading
        # ----------------------------------------------------

        if line.startswith("### "):

            heading = line[4:].strip()

            story.append(
                Paragraph(
                    escape_html(heading),
                    subheading_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Horizontal line
        # ----------------------------------------------------

        if line == "---":

            story.append(
                Spacer(1, 6)
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Markdown table
        # ----------------------------------------------------

        if line.startswith("|"):

            table_lines = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
            ):

                current = lines[i].strip()

                # Ignore markdown separator row
                if not re.match(
                    r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                    current
                ):

                    table_lines.append(
                        current
                    )

                i += 1

            if table_lines:

                table_data = []

                for table_line in table_lines:

                    cells = [
                        cell.strip()
                        for cell in table_line.strip("|").split("|")
                    ]

                    table_data.append(
                        [
                            Paragraph(
                                escape_html(cell),
                                body_style
                            )
                            for cell in cells
                        ]
                    )

                if table_data:

                    table = Table(
                        table_data,
                        repeatRows=1,
                        colWidths=None,
                        hAlign="LEFT"
                    )

                    table.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.lightgrey
                                ),
                                (
                                    "TEXTCOLOR",
                                    (0, 0),
                                    (-1, 0),
                                    colors.black
                                ),
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.5,
                                    colors.grey
                                ),
                                (
                                    "VALIGN",
                                    (0, 0),
                                    (-1, -1),
                                    "TOP"
                                ),
                                (
                                    "LEFTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6
                                ),
                                (
                                    "RIGHTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6
                                ),
                                (
                                    "TOPPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    5
                                ),
                                (
                                    "BOTTOMPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    5
                                )
                            ]
                        )
                    )

                    story.append(
                        KeepTogether(
                            table
                        )
                    )

                    story.append(
                        Spacer(1, 8)
                    )

            continue

        # ----------------------------------------------------
        # Bullet point
        # ----------------------------------------------------

        if (
            line.startswith("- ")
            or line.startswith("* ")
        ):

            bullet_text = line[2:].strip()

            story.append(
                Paragraph(
                    "• "
                    + markdown_to_html(
                        bullet_text
                    ),
                    bullet_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Numbered list
        # ----------------------------------------------------

        numbered_match = re.match(
            r"^(\d+)\.\s+(.*)",
            line
        )

        if numbered_match:

            number = numbered_match.group(1)

            text = numbered_match.group(2)

            story.append(
                Paragraph(
                    f"{number}. "
                    + markdown_to_html(text),
                    bullet_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Important disclaimer
        # ----------------------------------------------------

        if (
            line.startswith("**IMPORTANT")
            or line.startswith("IMPORTANT")
        ):

            story.append(
                Paragraph(
                    markdown_to_html(line),
                    disclaimer_style
                )
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        story.append(
            Paragraph(
                markdown_to_html(line),
                body_style
            )
        )

        i += 1

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    def add_footer(canvas, document):

        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            7
        )

        canvas.drawCentredString(
            A4[0] / 2,
            10 * mm,
            "Synthetic Medical Report | "
            "Educational/Research Use Only"
        )

        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {document.page}"
        )

        canvas.restoreState()

    # --------------------------------------------------------
    # Build PDF
    # --------------------------------------------------------

    doc.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    return filepath


# ============================================================
# MARKDOWN → REPORTLAB HTML
# ============================================================

def markdown_to_html(text):
    """
    Convert basic Markdown formatting into
    ReportLab-compatible HTML.
    """

    text = escape_html(text)

    # Bold
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"<b>\1</b>",
        text
    )

    # Italic
    text = re.sub(
        r"\*(.*?)\*",
        r"<i>\1</i>",
        text
    )

    # Inline code
    text = re.sub(
        r"`(.*?)`",
        r"<font name='Courier'>\1</font>",
        text
    )

    return text


# ============================================================
# HTML ESCAPING
# ============================================================

def escape_html(text):
    """
    Escape characters that can interfere with
    ReportLab's paragraph parser.
    """

    if text is None:

        return ""

    text = str(text)

    text = text.replace(
        "&",
        "&amp;"
    )

    text = text.replace(
        "<",
        "&lt;"
    )

    text = text.replace(
        ">",
        "&gt;"
    )

    return text