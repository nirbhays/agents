"""
Generate a professional PDF document from the DevOps certification questions.
Enhanced with modern design, better typography, and comprehensive explanations.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Table, TableStyle, KeepTogether, Image, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import re


class NumberedCanvas(canvas.Canvas):
    """Canvas with page numbers and footer."""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        page = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(7.5*inch, 0.5*inch, page)
        # Footer line
        self.setStrokeColor(colors.HexColor('#dadce0'))
        self.setLineWidth(0.5)
        self.line(0.75*inch, 0.6*inch, 7.5*inch, 0.6*inch)


def parse_devops_questions(file_path):
    """Parse the DevOps.txt file and extract questions with answers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    
    # Split by question markers - handle both ### Question # and Question #:
    sections = re.split(r'(?=###\s+Question\s+#|^Question\s+#[:\s])', content, flags=re.MULTILINE)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        question_data = {
            'title': '',
            'scenario': '',
            'question': '',
            'options': [],
            'correct_answer': '',
            'hint': '',
            'topic': ''
        }
        
        in_scenario = False
        in_question = False
        in_options = False
        found_first_text = False  # Track if we've seen content after title
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Question title - handle both ### Question # and Question #:
            if ((line.startswith('###') or line.startswith('Question #')) and 
                ('Question' in line or 'question' in line.lower()) and 
                not question_data['title']):
                question_data['title'] = line.replace('###', '').strip()
                # Check next line for topic
                if i + 1 < len(lines) and 'Topic' in lines[i+1]:
                    question_data['topic'] = lines[i+1].strip()
                continue
            
            # Topic on same or next line
            if line.startswith('Topic #:') or line.startswith('Topic'):
                question_data['topic'] = line
                continue
                
            # Scenario marker
            if line.startswith('**Scenario:**'):
                in_scenario = True
                found_first_text = True
                question_data['scenario'] = line.replace('**Scenario:**', '').strip()
                continue
            
            # Question marker
            if line.startswith('**Question:**'):
                in_scenario = False
                in_question = True
                question_data['question'] = line.replace('**Question:**', '').strip()
                continue
            
            # If no explicit markers, treat text before options as question
            if not found_first_text and line and not line.startswith('---') and not line.startswith('[All Professional'):
                in_question = True
                found_first_text = True
                question_data['question'] = line
                continue
            
            # Options - more patterns
            if (re.match(r'^\*\s+\*\*[A-D]\.\*\*', line) or 
                re.match(r'^[A-D]\.\s+', line) or
                re.match(r'^[A-D]\.\s*•', line)):
                in_question = False
                in_options = True
                # Clean up option formatting
                option = re.sub(r'^\*\s+\*\*([A-D])\.\*\*', r'\1.', line)
                option = re.sub(r'^([A-D])\.\s*•', r'\1.', option)
                question_data['options'].append(option)
                continue
            
            # Correct answer - multiple patterns
            if (line.startswith('*   **Correct Answer') or 
                line.startswith('Solution:') or
                line.startswith('Correct Answer:') or
                line.startswith('**Correct Answer')):
                correct = re.sub(r'\*\s+\*\*Correct Answer[s]?:\*\*', '', line)
                correct = re.sub(r'Solution:', '', correct).strip()
                correct = re.sub(r'Correct Answer[s]?:', '', correct).strip()
                correct = re.sub(r'\*\*', '', correct).strip()
                # Extract just the letter(s) - handle "Most Voted" and other annotations
                correct = re.sub(r'\s*Most Voted.*$', '', correct, flags=re.IGNORECASE)
                correct_match = re.search(r'([A-D](?:\s*[&,]\s*[A-D])*)', correct)
                if correct_match:
                    question_data['correct_answer'] = correct_match.group(1).replace(',', ' &')
                continue
            
            # Hint
            if line.startswith('*   **Hint:**') or line.startswith('**Hint:**'):
                question_data['hint'] = line.replace('*   **Hint:**', '').replace('**Hint:**', '').strip()
                continue
            
            # Continue building current section
            if in_scenario and line and not line.startswith('---') and not line.startswith('**Question:**'):
                question_data['scenario'] += ' ' + line
            elif in_question and line and not line.startswith('---') and not re.match(r'^[A-D]\.', line):
                question_data['question'] += ' ' + line
            elif in_options and line and not line.startswith('---') and not line.startswith('*   **') and not line.startswith('Solution:') and not line.startswith('Question #'):
                if question_data['options'] and not re.match(r'^[A-D]\.', line):
                    question_data['options'][-1] += ' ' + line
        
        # Only add if we have meaningful content
        if question_data['title'] and (question_data['scenario'] or question_data['question'] or question_data['options']):
            questions.append(question_data)
    
    return questions


def create_pdf(input_file, output_file):
    """Create a highly professional PDF from the DevOps questions."""
    
    # Parse questions
    questions = parse_devops_questions(input_file)
    
    # Create PDF document with custom canvas
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # ========== ENHANCED PROFESSIONAL STYLES ==========
    
    # Title page styles
    main_title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=32,
        textColor=colors.HexColor('#1967D2'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=38
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.HexColor('#5F6368'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#80868B'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Question styles - minimalist
    question_number_style = ParagraphStyle(
        'QuestionNumber',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#202124'),
        spaceAfter=0,
        spaceBefore=0,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT,
        borderPadding=0,
        leading=14
    )
    
    question_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#202124'),
        spaceAfter=10,
        spaceBefore=8,
        fontName='Helvetica-Bold',
        leftIndent=5
    )
    
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=3,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#5F6368'),
        leftIndent=0
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        textColor=colors.HexColor('#202124'),
        leftIndent=0,
        rightIndent=0,
        leading=12
    )
    
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        leftIndent=15,
        fontName='Helvetica',
        textColor=colors.HexColor('#202124'),
        bulletIndent=0,
        leading=12
    )
    
    correct_answer_style = ParagraphStyle(
        'CorrectAnswer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#0F9D58'),
        spaceAfter=6,
        fontName='Helvetica-Bold',
        backColor=colors.white,
        borderPadding=0,
        borderRadius=0,
        leftIndent=0
    )
    
    explanation_style = ParagraphStyle(
        'ExplanationStyle',
        parent=styles['Normal'],
        fontSize=9.5,
        textColor=colors.HexColor('#5F6368'),
        spaceAfter=8,
        leftIndent=0,
        rightIndent=0,
        fontName='Helvetica-Oblique',
        backColor=colors.white,
        borderColor=colors.white,
        borderWidth=0,
        borderPadding=0,
        borderRadius=0,
        leading=11
    )
    
    appendix_title_style = ParagraphStyle(
        'AppendixTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1967D2'),
        spaceAfter=25,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1967D2'),
        spaceAfter=12,
        spaceBefore=15,
        fontName='Helvetica-Bold',
        borderColor=colors.HexColor('#1967D2'),
        borderWidth=0,
        leftIndent=0
    )
    
    # ========== TITLE PAGE ==========
    elements.append(Spacer(1, 1.5*inch))
    
    # Main title - minimalist
    elements.append(Paragraph("Google Cloud", main_title_style))
    elements.append(Paragraph("Professional DevOps Engineer", main_title_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(HRFlowable(width="30%", thickness=2, color=colors.HexColor('#1967D2'), spaceAfter=25, spaceBefore=0, hAlign='CENTER'))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Certification Practice Examination", subtitle_style))
    elements.append(Paragraph("Complete Question Bank with Detailed Explanations", subtitle_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Document Generated: {datetime.now().strftime('%B %d, %Y')}", date_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Professional summary table
    summary_data = [
        ['', ''],
        ['Total Questions', str(len(questions))],
        ['Question Format', 'Multiple Choice (Single & Multiple Answer)'],
        ['Difficulty Level', 'Professional Certification Standard'],
        ['Coverage', 'Complete DevOps Engineer Exam Topics'],
        ['', '']
    ]
    
    summary_table = Table(summary_data, colWidths=[2.8*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 1), (-1, -2), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -2), colors.HexColor('#202124')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (0, -2), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, -2), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -2), 10),
        ('LINEBELOW', (0, 1), (-1, -2), 0.5, colors.HexColor('#E8E8E8')),
        ('VALIGN', (0, 1), (-1, -2), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -2), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -2), 8),
        ('LEFTPADDING', (0, 1), (-1, -2), 0)
    ]))
    
    elements.append(summary_table)
    elements.append(PageBreak())
    
    # ========== QUESTIONS SECTION ==========
    for idx, q in enumerate(questions, 1):
        question_elements = []
        
        # Clean question title extraction - avoid duplicate numbering
        title_text = q['title'].replace('Question #:', '').replace('Question #', '').replace('Topic #: 1', '').replace('Topic #:', '').strip()
        
        # Remove leading/trailing numbers and colons that match the question index
        title_parts = title_text.split(':')
        if len(title_parts) > 0:
            first_part = title_parts[0].strip()
            # If first part is just the question number, remove it
            if first_part.isdigit() and first_part == str(idx):
                if len(title_parts) > 1:
                    title_text = ':'.join(title_parts[1:]).strip()
                else:
                    title_text = ''
        
        # Final check: if title is empty or just a number matching index, don't display it
        if title_text and title_text != str(idx) and not (title_text.isdigit() and int(title_text) == idx):
            display_title = f"Question {idx}: {title_text}"
        else:
            display_title = f"Question {idx}"
        
        # Question number header - minimalist style
        question_elements.append(Paragraph(display_title, question_number_style))
        question_elements.append(Spacer(1, 0.08*inch))
        
        # Scenario section
        if q['scenario']:
            question_elements.append(Paragraph("<b>Scenario</b>", section_header_style))
            question_elements.append(Paragraph(q['scenario'], body_style))
            question_elements.append(Spacer(1, 0.06*inch))
        
        # Question section
        if q['question']:
            question_elements.append(Paragraph("<b>Question</b>", section_header_style))
            question_elements.append(Paragraph(q['question'], body_style))
            question_elements.append(Spacer(1, 0.06*inch))
        
        # Answer options - minimalist formatting
        if q['options']:
            for option in q['options']:
                # Highlight correct answer option
                option_letter = option[0] if len(option) > 0 else ''
                if q['correct_answer'] and option_letter in q['correct_answer']:
                    formatted_option = f"<b><font color='#0F9D58'>{option}</font></b>"
                else:
                    formatted_option = option
                question_elements.append(Paragraph(formatted_option, option_style))
            question_elements.append(Spacer(1, 0.08*inch))
        
        # Correct answer - minimal box
        if q['correct_answer']:
            question_elements.append(Paragraph(
                f"✓ Correct Answer: {q['correct_answer']}", 
                correct_answer_style
            ))
        
        # Explanation
        if q['hint']:
            question_elements.append(Paragraph(
                f"<b>Explanation:</b> {q['hint']}", 
                explanation_style
            ))
        
        # Minimal separator
        question_elements.append(Spacer(1, 0.1*inch))
        question_elements.append(HRFlowable(
            width="100%", 
            thickness=0.5, 
            color=colors.HexColor('#E8E8E8'), 
            spaceAfter=0
        ))
        
        # Keep question together
        elements.append(KeepTogether(question_elements))
        elements.append(Spacer(1, 0.15*inch))
        
        # Strategic page breaks
        if idx % 2 == 0:
            elements.append(PageBreak())
    
    # ========== COMPREHENSIVE EXPLANATION APPENDIX ==========
    elements.append(PageBreak())
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Understanding the Correct Answers", appendix_title_style))
    elements.append(HRFlowable(width="30%", thickness=2, color=colors.HexColor('#1967D2'), spaceAfter=25, hAlign='CENTER'))
    
    intro_text = """
    <b>Purpose of This Section:</b> This comprehensive guide explains the reasoning behind each correct 
    answer in the practice examination. Understanding these explanations is essential for mastering 
    Google Cloud DevOps Engineer concepts and achieving certification success. Each explanation is 
    designed to reinforce best practices and help you recognize patterns in real-world scenarios.
    """
    elements.append(Paragraph(intro_text, body_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Key Concepts
    elements.append(Paragraph("Core Competency Areas", section_title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    key_concepts = [
        ("<b>1. Observability & Monitoring</b>", 
         "Master Cloud Operations (formerly Stackdriver) suite including Cloud Monitoring for metrics and alerting, "
         "Cloud Logging for centralized log management, Cloud Trace for distributed tracing across microservices, "
         "and Cloud Profiler for performance analysis. Understand when to use each tool and how they integrate."),
        
        ("<b>2. Site Reliability Engineering (SRE) Principles</b>", 
         "Implement SRE best practices including blameless postmortems focused on system improvements, "
         "defining and measuring Service Level Indicators (SLIs) and Objectives (SLOs), managing error budgets, "
         "and creating automation to reduce toil. Always prioritize learning and sharing knowledge broadly."),
        
        ("<b>3. CI/CD Pipeline Optimization</b>", 
         "Design robust continuous integration and deployment pipelines using Cloud Build, integrate automated "
         "testing at multiple stages, implement progressive deployment strategies (canary, blue/green, rolling), "
         "and incorporate monitoring and rollback mechanisms. Minimize manual intervention."),
        
        ("<b>4. Kubernetes & Container Orchestration</b>", 
         "Leverage Google Kubernetes Engine (GKE) for managed Kubernetes with features like cluster autoscaling, "
         "node auto-repair, and workload identity. Use sidecar patterns for cross-cutting concerns, implement "
         "proper resource limits, and integrate with Google Cloud services securely."),
        
        ("<b>5. Security & Identity Management</b>", 
         "Apply the principle of least privilege consistently, use service accounts for workload identity, "
         "implement proper IAM role bindings, avoid long-lived credentials, enable security scanning, "
         "and use organization policies for governance at scale."),
        
        ("<b>6. Service Level Objectives & Indicators</b>", 
         "Define meaningful SLIs that reflect user experience (latency, availability, throughput), calculate "
         "SLIs correctly (e.g., requests meeting target / total requests), use appropriate percentiles "
         "(p50, p95, p99) for latency measurements, and set realistic SLOs based on business requirements."),
        
        ("<b>7. Logging Architecture & Strategy</b>", 
         "Utilize automatic log collection from stdout/stderr in containerized environments, implement "
         "structured logging with JSON for better querying, use log-based metrics for monitoring, "
         "configure appropriate retention policies, and implement log sinks for long-term storage."),
        
        ("<b>8. Infrastructure as Code & Automation</b>", 
         "Use Terraform or Cloud Deployment Manager for infrastructure provisioning, implement GitOps "
         "workflows, leverage Cloud Build for automated deployments, use Cloud Pub/Sub for event-driven "
         "automation, and maintain infrastructure code in version control with proper review processes."),
        
        ("<b>9. Cost Optimization & Resource Management</b>", 
         "Implement autoscaling to match demand, use appropriate machine types and committed use discounts, "
         "leverage preemptible VMs for fault-tolerant workloads, implement resource quotas and budgets, "
         "and regularly review and optimize resource utilization."),
        
        ("<b>10. Incident Management & Response</b>", 
         "Establish clear incident response procedures, implement effective alerting with proper thresholds, "
         "maintain runbooks for common scenarios, conduct blameless postmortems after incidents, "
         "and continuously improve systems based on lessons learned.")
    ]
    
    for title, content in key_concepts:
        elements.append(Paragraph(title, question_title_style))
        elements.append(Paragraph(content, body_style))
        elements.append(Spacer(1, 0.12*inch))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Why Answers Are Correct
    elements.append(Paragraph("Answer Selection Criteria", section_title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    criteria_text = """
    <b>Each correct answer demonstrates adherence to Google Cloud best practices:</b><br/><br/>
    
    <b>✓ Principle of Least Privilege:</b> Grant minimum necessary permissions, use fine-grained IAM roles, 
    and avoid overly permissive access.<br/><br/>
    
    <b>✓ Automation Over Manual Process:</b> Prefer automated solutions that reduce human error and toil, 
    use managed services, and implement self-healing systems.<br/><br/>
    
    <b>✓ Managed Services First:</b> Leverage Google-managed services (GKE, Cloud Build, Cloud Run) rather 
    than self-managed alternatives to reduce operational overhead.<br/><br/>
    
    <b>✓ Scalability & Reliability:</b> Solutions must scale automatically, handle failures gracefully, 
    and maintain service availability during issues.<br/><br/>
    
    <b>✓ Cost Efficiency:</b> Balance performance with cost, use appropriate resource sizing, and implement 
    autoscaling to optimize spend.<br/><br/>
    
    <b>✓ Security by Design:</b> Implement defense in depth, encrypt data in transit and at rest, 
    use private networking, and audit access regularly.<br/><br/>
    
    <b>✓ Observability:</b> Comprehensive monitoring, logging, and tracing must be integrated throughout 
    the system architecture.<br/><br/>
    
    <b>✓ Minimal Development Effort:</b> Use existing tools, services, and integrations rather than 
    building custom solutions.<br/><br/>
    
    <b>Common Reasons Incorrect Options Are Rejected:</b><br/><br/>
    
    ✗ <b>Overly Permissive:</b> Grants excessive permissions violating least privilege<br/>
    ✗ <b>Manual Intervention Required:</b> Requires human involvement for routine operations<br/>
    ✗ <b>Poor Scalability:</b> Cannot handle increasing load efficiently<br/>
    ✗ <b>Unnecessary Complexity:</b> Over-engineered when simpler solutions exist<br/>
    ✗ <b>Cost Inefficient:</b> Wastes resources or uses expensive approaches unnecessarily<br/>
    ✗ <b>Incomplete Solution:</b> Doesn't fully address the stated requirements<br/>
    ✗ <b>Security Gaps:</b> Introduces security vulnerabilities or compliance issues<br/>
    ✗ <b>Anti-patterns:</b> Violates established best practices or SRE principles
    """
    elements.append(Paragraph(criteria_text, body_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Study Strategy
    elements.append(Paragraph("Recommended Study Approach", section_title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    strategy_text = """
    <b>1. Comprehensive Review:</b> For each question, study both the correct answer AND why other options 
    are incorrect. This develops pattern recognition skills.<br/><br/>
    
    <b>2. Hands-On Practice:</b> Implement solutions in a GCP sandbox environment. Practical experience 
    reinforces theoretical knowledge.<br/><br/>
    
    <b>3. Documentation Deep-Dive:</b> Reference official Google Cloud documentation for each service 
    mentioned. Understand service capabilities and limitations.<br/><br/>
    
    <b>4. Scenario-Based Learning:</b> Focus on understanding WHY solutions work in specific contexts 
    rather than memorizing answers.<br/><br/>
    
    <b>5. Best Practices First:</b> Internalize Google-recommended best practices and architectural 
    patterns across all service areas.<br/><br/>
    
    <b>6. Real-World Application:</b> Consider how each concept applies to production environments 
    and enterprise-scale deployments.<br/><br/>
    
    <b>Success Tip:</b> The Professional Cloud DevOps Engineer certification validates your ability to 
    implement technical solutions and operational procedures. Focus on understanding the reasoning behind 
    architectural decisions, not just the technical implementation details.
    """
    elements.append(Paragraph(strategy_text, body_style))
    
    # Build PDF with custom canvas for page numbers
    doc.build(elements, canvasmaker=NumberedCanvas)
    
    print(f"\n{'='*60}")
    print(f"✓ PDF Generated Successfully!")
    print(f"{'='*60}")
    print(f"Output File: {output_file}")
    print(f"Total Questions: {len(questions)}")
    print(f"Document Pages: Professional multi-page format")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    input_file = r"c:\Users\SIN3WZ\Documents\Learn\Certs\Devops.txt"
    output_file = r"c:\Users\SIN3WZ\Documents\Learn\Certs\GCP_DevOps_Practice_Questions.pdf"
    
    print("Generating DevOps Practice Questions PDF...")
    create_pdf(input_file, output_file)
    print("Done!")
