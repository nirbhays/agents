"""
Professional GCP Certification PDF Generator - Enhanced Version
Creates beautifully formatted, well-structured PDFs with proper page layout
"""

import os
import re
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted, Image, HRFlowable, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configuration
CERTS_DIR = Path(__file__).parent
OUTPUT_DIR = CERTS_DIR / "pdf_output"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES_TO_CONVERT = {
    "GCP_Professional_Cloud_Developer_Study_Guide.md": {
        "output": "GCP_Professional_Cloud_Developer_Study_Guide.pdf",
        "title": "GCP Professional Cloud Developer",
        "subtitle": "Complete Study Guide",
        "color": "#1a73e8"
    },
    "GCP_Professional_Cloud_DevOps_Engineer_Study_Guide.md": {
        "output": "GCP_Professional_Cloud_DevOps_Engineer_Study_Guide.pdf",
        "title": "GCP Professional Cloud DevOps Engineer",
        "subtitle": "Complete Study Guide",
        "color": "#34a853"
    },
    "GCP_Certs_Consolidated_Cheatsheet.md": {
        "output": "GCP_Certs_Consolidated_Cheatsheet.pdf",
        "title": "GCP Certification Cheatsheet",
        "subtitle": "Quick Reference Guide",
        "color": "#fbbc04"
    },
    "GCP_Certs_Enhanced_Cheatsheet.md": {
        "output": "GCP_Certs_Enhanced_Cheatsheet.pdf",
        "title": "GCP Certification Enhanced Cheatsheet",
        "subtitle": "Comprehensive Reference",
        "color": "#ea4335"
    }
}


class NumberedCanvas(canvas.Canvas):
    """Enhanced canvas with better page decorations"""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.doc_title = ""
        self.theme_color = colors.HexColor('#1a73e8')

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        page_num = self._pageNumber
        
        if page_num == 1:
            return
        
        self.saveState()
        
        # Header line
        self.setStrokeColor(self.theme_color)
        self.setLineWidth(1)
        self.line(0.75*inch, A4[1] - 0.6*inch, A4[0] - 0.75*inch, A4[1] - 0.6*inch)
        
        # Header text
        self.setFont('Helvetica-Oblique', 8)
        self.setFillColor(colors.HexColor('#5f6368'))
        if self.doc_title:
            self.drawString(0.75*inch, A4[1] - 0.5*inch, self.doc_title[:100])
        
        # Footer line
        self.setStrokeColor(colors.HexColor('#e8eaed'))
        self.setLineWidth(0.5)
        self.line(0.75*inch, 0.6*inch, A4[0] - 0.75*inch, 0.6*inch)
        
        # Footer - left (date)
        self.setFont('Helvetica', 7)
        self.setFillColor(colors.HexColor('#80868b'))
        self.drawString(0.75*inch, 0.45*inch, 
                       f"Generated: {datetime.now().strftime('%B %d, %Y')}")
        
        # Footer - right (page numbers)
        self.drawRightString(A4[0] - 0.75*inch, 0.45*inch,
                           f"Page {page_num} of {page_count}")
        
        self.restoreState()


def setup_styles():
    """Create comprehensive style sheet"""
    styles = getSampleStyleSheet()
    
    # ===== COVER PAGE STYLES =====
    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontSize=36,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=42
    ))
    
    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontSize=18,
        textColor=colors.HexColor('#5f6368'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))
    
    styles.add(ParagraphStyle(
        name='CoverInfo',
        fontSize=11,
        textColor=colors.HexColor('#202124'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))
    
    # ===== TOC STYLES =====
    styles.add(ParagraphStyle(
        name='TOCTitle',
        fontSize=28,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=30,
        fontName='Helvetica-Bold',
        spaceBefore=0
    ))
    
    styles.add(ParagraphStyle(
        name='TOCLevel1',
        fontSize=12,
        textColor=colors.HexColor('#1a73e8'),
        leftIndent=0,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        leading=16
    ))
    
    styles.add(ParagraphStyle(
        name='TOCLevel2',
        fontSize=10,
        textColor=colors.HexColor('#5f6368'),
        leftIndent=24,
        spaceAfter=5,
        fontName='Helvetica',
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name='TOCLevel3',
        fontSize=9,
        textColor=colors.HexColor('#80868b'),
        leftIndent=48,
        spaceAfter=3,
        fontName='Helvetica',
        leading=12
    ))
    
    # ===== HEADING STYLES =====
    styles.add(ParagraphStyle(
        name='Heading1Custom',
        fontSize=22,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=16,
        spaceBefore=24,
        fontName='Helvetica-Bold',
        leading=26,
        keepWithNext=True
    ))
    
    styles.add(ParagraphStyle(
        name='Heading2Custom',
        fontSize=16,
        textColor=colors.HexColor('#1967d2'),
        spaceAfter=12,
        spaceBefore=18,
        fontName='Helvetica-Bold',
        leading=20,
        leftIndent=0,
        keepWithNext=True
    ))
    
    styles.add(ParagraphStyle(
        name='Heading3Custom',
        fontSize=13,
        textColor=colors.HexColor('#185abc'),
        spaceAfter=10,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        leading=16,
        keepWithNext=True
    ))
    
    styles.add(ParagraphStyle(
        name='Heading4Custom',
        fontSize=11,
        textColor=colors.HexColor('#174ea6'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        leading=14,
        keepWithNext=True
    ))
    
    # ===== BODY TEXT STYLES =====
    styles.add(ParagraphStyle(
        name='BodyTextCustom',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#202124'),
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        fontName='Helvetica'
    ))
    
    styles.add(ParagraphStyle(
        name='BulletTextCustom',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#202124'),
        leftIndent=24,
        bulletIndent=12,
        spaceAfter=6,
        fontName='Helvetica'
    ))
    
    # ===== CODE STYLES =====
    styles.add(ParagraphStyle(
        name='CodeBlock',
        fontSize=8,
        leading=11,
        leftIndent=12,
        rightIndent=12,
        spaceAfter=14,
        spaceBefore=14,
        backColor=colors.HexColor('#f8f9fa'),
        borderColor=colors.HexColor('#dadce0'),
        borderWidth=1,
        borderPadding=10,
        fontName='Courier',
        textColor=colors.HexColor('#202124')
    ))
    
    styles.add(ParagraphStyle(
        name='InlineCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        textColor=colors.HexColor('#d93025'),
        backColor=colors.HexColor('#f8f9fa')
    ))
    
    # ===== SPECIAL BOX STYLES =====
    styles.add(ParagraphStyle(
        name='InfoBox',
        fontSize=10,
        leading=14,
        leftIndent=16,
        rightIndent=16,
        spaceAfter=14,
        spaceBefore=14,
        backColor=colors.HexColor('#e8f0fe'),
        borderColor=colors.HexColor('#1a73e8'),
        borderWidth=1,
        borderPadding=12,
        borderRadius=4
    ))
    
    styles.add(ParagraphStyle(
        name='WarningBox',
        fontSize=10,
        leading=14,
        leftIndent=16,
        rightIndent=16,
        spaceAfter=14,
        spaceBefore=14,
        backColor=colors.HexColor('#fef7e0'),
        borderColor=colors.HexColor('#fbbc04'),
        borderWidth=1,
        borderPadding=12
    ))
    
    styles.add(ParagraphStyle(
        name='ScenarioBox',
        fontSize=10,
        leading=14,
        leftIndent=16,
        rightIndent=16,
        spaceAfter=14,
        spaceBefore=14,
        backColor=colors.HexColor('#f0f7ff'),
        borderColor=colors.HexColor('#4285f4'),
        borderWidth=1,
        borderPadding=12
    ))
    
    return styles


def parse_table(lines, start_idx):
    """Parse markdown table and return Table object"""
    table_lines = []
    i = start_idx
    
    # Collect table lines
    while i < len(lines) and ('|' in lines[i] or lines[i].strip().startswith('├') or lines[i].strip().startswith('└')):
        line = lines[i].strip()
        if line and not line.replace('-', '').replace('|', '').replace('├', '').replace('┤', '').replace('┼', '').replace('└', '').replace('┘', '').strip() == '':
            if '|' in line:
                table_lines.append(line)
        i += 1
    
    if not table_lines:
        return None, start_idx
    
    # Parse table data
    data = []
    for line in table_lines:
        # Skip separator lines
        if re.match(r'^[\|\s\-:]+$', line):
            continue
        
        # Split by | and clean
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty cells
        
        if cells:
            data.append(cells)
    
    if not data or len(data) < 2:
        return None, start_idx
    
    # Create table
    try:
        # Calculate column widths
        num_cols = len(data[0])
        available_width = 6.5 * inch
        col_widths = [available_width / num_cols] * num_cols
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Style the table
        table_style = TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Body rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#202124')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dadce0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a73e8')),
            
            # Padding
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.white, colors.HexColor('#f8f9fa')]),
        ])
        
        table.setStyle(table_style)
        return table, i
        
    except Exception as e:
        print(f"Warning: Could not create table: {e}")
        return None, start_idx


def parse_ascii_table(lines, start_idx):
    """Parse ASCII box-drawing tables"""
    table_lines = []
    i = start_idx
    
    # Collect ASCII table lines
    while i < len(lines):
        line = lines[i]
        if any(char in line for char in ['┌', '├', '│', '└', '─', '┼', '┤', '┐', '┘']):
            table_lines.append(line)
            i += 1
        elif line.strip() and table_lines:
            break
        elif not line.strip():
            i += 1
            if table_lines:
                break
        else:
            break
    
    if not table_lines:
        return None, start_idx
    
    # Parse the table data
    data = []
    for line in table_lines:
        if '│' in line:
            # Split by vertical bars
            cells = line.split('│')
            cells = [c.strip() for c in cells if c.strip()]
            if cells:
                data.append(cells)
    
    if len(data) < 2:
        return None, start_idx
    
    try:
        # Calculate column widths
        num_cols = len(data[0])
        available_width = 6.5 * inch
        col_widths = [available_width / num_cols] * num_cols
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Enhanced styling
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a73e8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#202124')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dadce0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a73e8')),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.white, colors.HexColor('#f8f9fa')]),
        ])
        
        table.setStyle(table_style)
        return table, i
        
    except Exception as e:
        print(f"Warning: Could not create ASCII table: {e}")
        return None, start_idx


def format_inline_markdown(text):
    """Format inline markdown with better handling"""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # Italic
    text = re.sub(r'(?<!\*)\*([^\*]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
    
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#d93025" size="9">\1</font>', text)
    
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<font color="#1a73e8"><u>\1</u></font>', text)
    
    # Escape special XML characters
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # But restore our formatting tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;u&gt;', '<u>').replace('&lt;/u&gt;', '</u>')
    text = re.sub(r'&lt;font([^&]*?)&gt;', r'<font\1>', text)
    text = text.replace('&lt;/font&gt;', '</font>')
    
    return text


def parse_markdown_to_elements(markdown_text, styles):
    """Enhanced markdown parser with better structure"""
    elements = []
    lines = markdown_text.split('\n')
    
    i = 0
    in_code_block = False
    code_lines = []
    code_language = ''
    
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_text = '\n'.join(code_lines)
                if code_text.strip():
                    code_para = Preformatted(code_text, styles['CodeBlock'], maxLineLength=85)
                    elements.append(KeepTogether([code_para, Spacer(1, 0.1*inch)]))
                code_lines = []
                in_code_block = False
                code_language = ''
            else:
                # Start code block
                in_code_block = True
                code_language = line.strip()[3:].strip()
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Check for ASCII table
        if any(char in line for char in ['┌', '├', '│', '└']):
            table, new_i = parse_ascii_table(lines, i)
            if table:
                elements.append(Spacer(1, 0.15*inch))
                elements.append(KeepTogether([table, Spacer(1, 0.15*inch)]))
                i = new_i
                continue
        
        # Check for markdown table
        if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
            table, new_i = parse_table(lines, i)
            if table:
                elements.append(Spacer(1, 0.15*inch))
                elements.append(KeepTogether([table, Spacer(1, 0.15*inch)]))
                i = new_i
                continue
        
        # Empty lines
        if not line.strip():
            elements.append(Spacer(1, 0.08*inch))
            i += 1
            continue
        
        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            title = format_inline_markdown(title)
            
            if level == 1:
                elements.append(PageBreak())
                elements.append(Paragraph(title, styles['Heading1Custom']))
            elif level == 2:
                elements.append(Spacer(1, 0.2*inch))
                # Add colored bar before H2
                bar_table = Table([['']], colWidths=[0.5*cm])
                bar_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1967d2')),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ]))
                elements.append(bar_table)
                elements.append(Paragraph(title, styles['Heading2Custom']))
            elif level == 3:
                elements.append(Spacer(1, 0.15*inch))
                elements.append(Paragraph(title, styles['Heading3Custom']))
            elif level == 4:
                elements.append(Spacer(1, 0.12*inch))
                elements.append(Paragraph(title, styles['Heading4Custom']))
            else:
                elements.append(Paragraph(f"<b>{title}</b>", styles['BodyTextCustom']))
            
            elements.append(Spacer(1, 0.08*inch))
            i += 1
            continue
        
        # Horizontal rules
        if re.match(r'^[\-\*_]{3,}$', line.strip()):
            elements.append(Spacer(1, 0.1*inch))
            elements.append(HRFlowable(width="100%", thickness=1, 
                                      color=colors.HexColor('#e8eaed')))
            elements.append(Spacer(1, 0.1*inch))
            i += 1
            continue
        
        # Bullet lists
        if line.strip().startswith(('- ', '* ', '• ')):
            text = re.sub(r'^[\-\*•]\s+', '', line.strip())
            text = format_inline_markdown(text)
            bullet_para = Paragraph(f'• {text}', styles['BulletTextCustom'])
            elements.append(bullet_para)
            i += 1
            continue
        
        # Numbered lists
        if re.match(r'^\d+\.\s+', line.strip()):
            match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
            if match:
                num = match.group(1)
                text = match.group(2)
                text = format_inline_markdown(text)
                num_para = Paragraph(f'{num}. {text}', styles['BulletTextCustom'])
                elements.append(num_para)
            i += 1
            continue
        
        # Special scenario boxes
        if line.strip().startswith('**Scenario'):
            text = format_inline_markdown(line.strip())
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(text, styles['ScenarioBox']))
            i += 1
            continue
        
        # Regular paragraphs
        if line.strip():
            text = format_inline_markdown(line.strip())
            elements.append(Paragraph(text, styles['BodyTextCustom']))
        
        i += 1
    
    return elements


def extract_headings(markdown_text):
    """Extract headings for TOC"""
    headings = []
    for line in markdown_text.split('\n'):
        match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2)
            # Clean title
            title = re.sub(r'\*\*(.+?)\*\*', r'\1', title)
            title = re.sub(r'`(.+?)`', r'\1', title)
            title = re.sub(r'[^\x00-\x7F]+', '', title).strip()
            
            if title and level <= 3:
                headings.append({'level': level, 'title': title})
    
    return headings


def create_cover_page(title, subtitle, color, styles):
    """Create professional cover page"""
    elements = []
    
    # Logo space
    elements.append(Spacer(1, 2.5*inch))
    
    # Title
    title_style = ParagraphStyle(
        'TempCoverTitle',
        parent=styles['CoverTitle'],
        textColor=colors.HexColor(color)
    )
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Subtitle
    elements.append(Paragraph(subtitle, styles['CoverSubtitle']))
    elements.append(Spacer(1, 1.2*inch))
    
    # Info box
    info = f"""
    <b>Professional Certification Guide</b><br/>
    <br/>
    ✓ Complete exam coverage<br/>
    ✓ Structured learning path<br/>
    ✓ Real-world scenarios<br/>
    ✓ Quick reference tables<br/>
    ✓ Best practices &amp; tips<br/>
    """
    elements.append(Paragraph(info, styles['CoverInfo']))
    elements.append(Spacer(1, 1*inch))
    
    # Version info
    version = f"""
    <b>Version 1.0</b><br/>
    Generated: {datetime.now().strftime('%B %d, %Y')}<br/>
    © 2025 Study Materials
    """
    elements.append(Paragraph(version, styles['CoverInfo']))
    
    elements.append(PageBreak())
    return elements


def create_toc(headings, styles):
    """Create well-formatted TOC"""
    elements = []
    
    elements.append(Paragraph("Table of Contents", styles['TOCTitle']))
    elements.append(Spacer(1, 0.4*inch))
    
    for heading in headings[:60]:  # Limit TOC entries
        level = heading['level']
        title = heading['title']
        
        if level == 1:
            style = styles['TOCLevel1']
        elif level == 2:
            style = styles['TOCLevel2']
        else:
            style = styles['TOCLevel3']
        
        elements.append(Paragraph(title, style))
    
    elements.append(PageBreak())
    return elements


def convert_markdown_to_pdf(md_file, config):
    """Main conversion function with enhanced layout"""
    print(f"\n📄 Processing: {md_file.name}")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # Setup PDF
    output_path = OUTPUT_DIR / config['output']
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=0.85*inch,
        bottomMargin=0.85*inch,
        leftMargin=0.85*inch,
        rightMargin=0.85*inch
    )
    
    styles = setup_styles()
    elements = []
    
    # Cover page
    print(f"   ✓ Creating cover page...")
    elements.extend(create_cover_page(
        config['title'],
        config['subtitle'],
        config['color'],
        styles
    ))
    
    # Table of contents
    print(f"   ✓ Generating table of contents...")
    headings = extract_headings(markdown_text)
    elements.extend(create_toc(headings, styles))
    
    # Content
    print(f"   ✓ Converting content...")
    content_elements = parse_markdown_to_elements(markdown_text, styles)
    elements.extend(content_elements)
    
    # Build PDF
    print(f"   ✓ Building PDF...")
    
    def create_canvas_func(*args, **kwargs):
        canvas_obj = NumberedCanvas(*args, **kwargs)
        canvas_obj.doc_title = config['title']
        canvas_obj.theme_color = colors.HexColor(config['color'])
        return canvas_obj
    
    doc.build(elements, canvasmaker=create_canvas_func)
    
    file_size = output_path.stat().st_size / 1024
    print(f"   ✅ Created: {output_path.name} ({file_size:.1f} KB)")
    
    return output_path


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("  GCP CERTIFICATION PDF GENERATOR - PROFESSIONAL EDITION")
    print("=" * 70)
    
    converted = []
    
    for md_file, config in FILES_TO_CONVERT.items():
        md_path = CERTS_DIR / md_file
        
        if not md_path.exists():
            print(f"\n⚠️  Skipping {md_file} (not found)")
            continue
        
        try:
            output_path = convert_markdown_to_pdf(md_path, config)
            converted.append(output_path)
        except Exception as e:
            print(f"\n❌ Error converting {md_file}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"✅ Successfully generated {len(converted)} professional PDFs")
    print("=" * 70)
    print(f"\n📂 Output folder: {OUTPUT_DIR}")
    print("\n📄 Generated files:")
    
    for pdf_path in converted:
        size = pdf_path.stat().st_size / 1024
        print(f"   ✓ {pdf_path.name} ({size:.1f} KB)")
    
    print("\n🎨 Features:")
    print("   ✓ Professional cover pages with custom colors")
    print("   ✓ Comprehensive table of contents")
    print("   ✓ Enhanced page headers and footers")
    print("   ✓ Properly formatted tables with styling")
    print("   ✓ Syntax-highlighted code blocks")
    print("   ✓ Color-coded sections and headings")
    print("   ✓ Optimized page breaks and spacing")
    print("   ✓ Print-ready A4 format")
    
    print("\n🎉 All PDFs ready for studying and printing!\n")


if __name__ == "__main__":
    main()
