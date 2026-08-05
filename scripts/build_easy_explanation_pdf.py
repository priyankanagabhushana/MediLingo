from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs/MediLingo_Easy_Explanation_Guide.md"
OUTPUT = PROJECT_ROOT / "output/pdf/MediLingo_Easy_Explanation_Guide.pdf"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("MediLingoSans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("MediLingoSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("MediLingoSans-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("MediLingoMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily(
    "MediLingoSans",
    normal="MediLingoSans",
    bold="MediLingoSans-Bold",
    italic="MediLingoSans-Oblique",
)

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT = 17 * mm
RIGHT = 17 * mm
TOP = 17 * mm
BOTTOM = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#0F766E")
PALE_TEAL = colors.HexColor("#E7F5F2")
PALE_BLUE = colors.HexColor("#EDF4FA")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6875")
GRID = colors.HexColor("#C9D4DE")

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="GuideTitle",
        parent=styles["Title"],
        fontName="MediLingoSans-Bold",
        fontSize=25,
        leading=30,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideSubtitle",
        parent=styles["Normal"],
        fontName="MediLingoSans",
        fontSize=12,
        leading=17,
        textColor=TEAL,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideH1",
        parent=styles["Heading1"],
        fontName="MediLingoSans-Bold",
        fontSize=18,
        leading=23,
        textColor=NAVY,
        spaceBefore=13,
        spaceAfter=8,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideH2",
        parent=styles["Heading2"],
        fontName="MediLingoSans-Bold",
        fontSize=13.5,
        leading=18,
        textColor=TEAL,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideH3",
        parent=styles["Heading3"],
        fontName="MediLingoSans-Bold",
        fontSize=10.8,
        leading=14,
        textColor=NAVY,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideBody",
        parent=styles["BodyText"],
        fontName="MediLingoSans",
        fontSize=8.8,
        leading=12.7,
        textColor=INK,
        spaceAfter=5,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideBodyTight",
        parent=styles["GuideBody"],
        fontSize=8.2,
        leading=11.2,
        spaceAfter=2,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideBullet",
        parent=styles["GuideBody"],
        leftIndent=12,
        firstLineIndent=-9,
        spaceAfter=2.5,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideNumber",
        parent=styles["GuideBody"],
        leftIndent=16,
        firstLineIndent=-16,
        spaceAfter=2.5,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideQuote",
        parent=styles["GuideBody"],
        fontName="MediLingoSans-Oblique",
        leftIndent=12,
        rightIndent=8,
        borderColor=TEAL,
        borderWidth=1,
        borderPadding=7,
        backColor=PALE_TEAL,
        spaceBefore=4,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="GuideMeta",
        parent=styles["GuideBody"],
        fontSize=8,
        leading=11,
        textColor=MUTED,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell",
        parent=styles["GuideBodyTight"],
        fontSize=6.7,
        leading=8.6,
        spaceAfter=0,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHeader",
        parent=styles["TableCell"],
        fontName="MediLingoSans-Bold",
        textColor=colors.white,
    )
)


def inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="MediLingoMono">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#0F766E"><u>\1</u></link>',
        escaped,
    )
    return escaped


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(lines: list[str]):
    rows = [split_table_row(line) for line in lines if line.strip()]
    if len(rows) < 2:
        return None
    if is_table_separator("|" + "|".join(rows[1]) + "|"):
        rows.pop(1)
    if not rows:
        return None
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["TableHeader"] if row_index == 0 else styles["TableCell"]
        data.append([Paragraph(inline(cell), style) for cell in row])
    widths = [CONTENT_WIDTH / columns] * columns
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.35)
    canvas.line(LEFT, 12 * mm, PAGE_WIDTH - RIGHT, 12 * mm)
    canvas.setFont("MediLingoSans", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, 7.4 * mm, "MediLingo easy explanation guide")
    canvas.drawRightString(PAGE_WIDTH - RIGHT, 7.4 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_story(markdown: str):
    lines = markdown.splitlines()
    story = []
    index = 0
    first_title = True
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line == "[[PAGEBREAK]]":
            story.append(PageBreak())
            index += 1
            continue
        if line == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=GRID, spaceBefore=2, spaceAfter=7))
            index += 1
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            if not first_title:
                story.append(PageBreak())
            story.append(Paragraph(inline(title), styles["GuideTitle"]))
            first_title = False
            index += 1
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading == "How to use this guide":
                story.append(PageBreak())
            story.append(Paragraph(inline(heading), styles["GuideH1"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:].strip()), styles["GuideH2"]))
            index += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(inline(line[5:].strip()), styles["GuideH3"]))
            index += 1
            continue
        if line.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].strip())
                index += 1
            story.append(Paragraph(inline(" ".join(quote_lines)), styles["GuideQuote"]))
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            table = make_table(table_lines)
            if table is not None:
                story.append(Spacer(1, 3))
                story.append(table)
                story.append(Spacer(1, 6))
            continue
        if line.startswith("- "):
            while index < len(lines) and lines[index].strip().startswith("- "):
                bullet = lines[index].strip()[2:].strip()
                story.append(Paragraph(inline("- " + bullet), styles["GuideBullet"]))
                index += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            while index < len(lines):
                match = re.match(r"^(\d+)\.\s+(.*)$", lines[index].strip())
                if not match:
                    break
                story.append(Paragraph(inline(f"{match.group(1)}. {match.group(2)}"), styles["GuideNumber"]))
                index += 1
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (
                not next_line
                or next_line == "---"
                or next_line.startswith(("# ", "## ", "### ", "#### ", ">", "|", "- "))
                or re.match(r"^\d+\.\s+", next_line)
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        text = " ".join(paragraph_lines)
        style = styles["GuideMeta"] if text.startswith(("Date of this snapshot:", "Project name shown", "Technical project directory")) else styles["GuideBody"]
        story.append(Paragraph(inline(text), style))
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    story = build_story(markdown)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=RIGHT,
        leftMargin=LEFT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="MediLingo: The Simple Story Behind the Project",
        author="MediLingo project",
        subject="Plain-language project guide for healthcare translation research",
        allowSplitting=1,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
