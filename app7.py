from flask import Flask, render_template, request, jsonify, render_template_string
import os
import sys
import traceback
import chromadb
from datetime import datetime
from evalution import evaluate_semantic_relevance
import io
import base64
import re

# ReportLab for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Import from model6
from model7 import (
    load_pdf,
    split_text_smart,
    create_chroma_db,
    load_chroma_collection,
    
    retrieve_similar_text_enhanced,
    format_retrieved_text_professionally,
    process_pdf_and_create_db,
    EnhancedEmbeddingFunction
)
from html.parser import HTMLParser

class PDFHTMLParser(HTMLParser):
    """Extracts text and basic formatting from HTML to build ReportLab flowables."""
    def __init__(self, story, heading1, heading2, heading3, normal, bullet):
        super().__init__()
        self.story = story
        self.h1_style = heading1
        self.h2_style = heading2
        self.h3_style = heading3
        self.normal_style = normal
        self.bullet_style = bullet
        self.current_text = []
        self.in_heading = False
        self.in_para = False
        self.in_list_item = False
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.in_heading = True
            self.current_text = []
        elif tag == 'p':
            self.in_para = True
            self.current_text = []
        elif tag == 'li':
            self.in_list_item = True
            self.current_text = []
        elif tag in ('b', 'strong'):
            self.current_text.append('<b>')
        elif tag in ('i', 'em'):
            self.current_text.append('<i>')
        # Add other inline tags as needed

    def handle_endtag(self, tag):
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            text = ''.join(self.current_text).strip()
            if text:
                if tag == 'h1':
                    self.story.append(Paragraph(text, self.h1_style))
                elif tag == 'h2':
                    self.story.append(Paragraph(text, self.h2_style))
                else:
                    self.story.append(Paragraph(text, self.h3_style))
            self.in_heading = False
            self.current_text = []
        elif tag == 'p':
            text = ''.join(self.current_text).strip()
            if text:
                self.story.append(Paragraph(text, self.normal_style))
            self.in_para = False
            self.current_text = []
        elif tag == 'li':
            text = ''.join(self.current_text).strip()
            if text:
                self.story.append(Paragraph(f'• {text}', self.bullet_style))
            self.in_list_item = False
            self.current_text = []
        elif tag in ('b', 'strong'):
            self.current_text.append('</b>')
        elif tag in ('i', 'em'):
            self.current_text.append('</i>')
        self.current_tag = None

    def handle_data(self, data):
        self.current_text.append(data)

    def handle_entityref(self, name):
        self.current_text.append(f'&{name};')

    def handle_charref(self, name):
        self.current_text.append(f'&#{name};')


def html_to_flowables(html_content, styles, h1_style, h2_style, h3_style, normal_style, bullet_style):
    """Parse HTML and return a list of ReportLab flowables."""
    story = []
    parser = PDFHTMLParser(story, h1_style, h2_style, h3_style, normal_style, bullet_style)
    parser.feed(html_content)
    parser.close()
    return story

app = Flask(__name__)

# =====================================================
# Configuration
# =====================================================
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DB_FOLDER'] = 'chroma_db'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['EXPORT_FOLDER'] = 'exports'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DB_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

# Global variables
database = None
full_pdf_text = ""
current_collection_name = "pdf_docs"
evaluation_questions = [
    "what is artificial intelligence",
    "What is Breadth First Search?",
    "Explain uninformed search",
    "How does heuristic search improve efficiency?",
    "Greedy Best First Search "
]

# Similarity threshold for out-of-document detection
# Adjusted for hybrid scoring (semantic + TF-IDF combination)
SIMILARITY_THRESHOLD = 0.2

# =====================================================
# Initialize Database
# =====================================================
def initialize_database():
    global database
    db_path = app.config['DB_FOLDER']
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        existing_collections = [c.name for c in client.list_collections()]
        
        if current_collection_name in existing_collections:
            print(f"📚 Loading existing collection: {current_collection_name}")
            database = load_chroma_collection(db_path, current_collection_name)
            try:
                count = database.count()
                print(f"✅ Collection loaded with {count} chunks")
            except:
                print("✅ Collection loaded")
        else:
            print("📚 No existing database found")
            database = None
    except Exception as e:
        print(f"⚠️ Error loading database: {e}")
        database = None
    
    return database

# Initialize database
print("\n🔄 Initializing database...")
database = initialize_database()

# =====================================================
# Routes
# =====================================================
@app.route('/')
def index():
    """Main page"""
    db_count = database.count() if database else 0
    return render_template_string(HTML_TEMPLATE, 
                                database=database is not None, 
                                db_count=db_count)

@app.route('/search', methods=['POST'])
def search():
    """Enhanced search endpoint with similarity threshold"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        n_results = int(data.get('n_results', 5))
        diversity = float(data.get('diversity', 0.3))
        
        if not query:
            return jsonify({'success': False, 'error': 'Empty query'})
        
        if not database:
            return jsonify({'success': False, 'error': 'No database initialized. Please upload a PDF first.'})
        
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Database has {database.count()} chunks")
        
        # Use enhanced retrieval with similarity threshold
        retrieved = retrieve_similar_text_enhanced(
            query, database, n_results, diversity_factor=diversity,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        
        print(f"📝 Retrieved {len(retrieved)} chunks above threshold {SIMILARITY_THRESHOLD}")
        
        if not retrieved:
            return jsonify({
                'success': False, 
                'error': '🔍 Information not found in the current document. Please try rephrasing your question or upload a different PDF.',
                'not_found': True
            })
        
        result = format_retrieved_text_professionally(retrieved, query)
        
        # Add confidence score
        result['metadata']['confidence'] = "High" if len(retrieved) >= n_results else "Medium"
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'File must be a PDF'})
        
        # Save file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"📄 Processing PDF: {file.filename}")

        global full_pdf_text
        full_pdf_text = load_pdf(filepath)

        global database
        collection = process_pdf_and_create_db(
            filepath, 
            app.config['DB_FOLDER'], 
            current_collection_name
        )
        
        # Remove temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if collection:
            database = collection
            count = collection.count()
            print(f"✅ Successfully created database with {count} chunks")
            
            return jsonify({
                'success': True, 
                'chunks': count,
                'message': f'Successfully processed PDF. Created {count} chunks.'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to process PDF'})
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    """Export search results as a nicely formatted PDF (extracts visible content)"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        query = data.get('query', 'Search Results')
        
        if not content:
            return jsonify({'success': False, 'error': 'No content to export'})
        
        # Remove <style> and <script> tags
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        heading2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        heading3_style = ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#0284c7'),
            spaceAfter=6,
            spaceBefore=6,
            fontName='Helvetica-Bold'
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        )
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            leftIndent=20,
            firstLineIndent=0,
            spaceAfter=4,
            textColor=colors.HexColor('#2c3e50'),
            bulletIndent=10,
            bulletFontName='Helvetica'
        )
        
        story = []
        story.append(Paragraph("IntelliVault - Search Results", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Query:</b> {query}", normal_style))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 24))
        
        # Convert HTML to flowables
        flowables = html_to_flowables(
            content, styles,
            heading1_style, heading2_style, heading3_style,
            normal_style, bullet_style
        )
        story.extend(flowables)
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'pdf': pdf_base64,
            'filename': f'intellivault_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        })
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
@app.route('/download-chat', methods=['POST'])
def download_chat():
    """Download chat history as a nicely formatted PDF"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'success': False, 'error': 'No messages to download'})
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ChatTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        query_style = ParagraphStyle(
            'QueryStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1e40af'),
            backColor=colors.HexColor('#e6f0ff'),
            borderPadding=10,
            borderWidth=1,
            borderColor=colors.HexColor('#93c5fd'),
            spaceAfter=10,
            leftIndent=20,
            rightIndent=20,
            fontName='Helvetica'
        )
        answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#065f46'),
            backColor=colors.HexColor('#e6f7e6'),
            borderPadding=10,
            borderWidth=1,
            borderColor=colors.HexColor('#86efac'),
            spaceAfter=15,
            leftIndent=20,
            rightIndent=20,
            fontName='Helvetica'
        )
        
        story = []
        story.append(Paragraph("IntelliVault Chat History", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        for msg in messages:
            # Strip HTML tags from message content for cleaner PDF
            clean_content = re.sub(r'<[^>]+>', ' ', msg['content'])
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            if msg['type'] == 'query':
                story.append(Paragraph(f"<b>Q:</b> {clean_content}", query_style))
            else:
                story.append(Paragraph(f"<b>A:</b> {clean_content}", answer_style))
            story.append(Spacer(1, 6))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'pdf': pdf_base64,
            'filename': f'intellivault_chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        })
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'database': database is not None,
        'chunks': database.count() if database else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats')
def stats():
    """Get database statistics"""
    if not database:
        return jsonify({'success': False, 'error': 'No database'})
    
    try:
        count = database.count()
        return jsonify({
            'success': True,
            'chunks': count,
            'collection': current_collection_name
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/evaluate', methods=['GET'])
def evaluate():
    global database, full_pdf_text
    if not database:
        return {"error": "Database not initialized"}
    report = evaluate_semantic_relevance(database, full_pdf_text, evaluation_questions, k=5)
    return report

# =====================================================
# Enhanced HTML Template with Lighter Professional Colors
# =====================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IntelliVault - Intelligent PDF Assistant</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* Light Professional Color Scheme */
        :root {
            --primary-50: #f0f9ff;
            --primary-100: #e0f2fe;
            --primary-200: #bae6fd;
            --primary-300: #7dd3fc;
            --primary-400: #38bdf8;
            --primary-500: #0ea5e9;
            --primary-600: #0284c7;
            --primary-700: #0369a1;
            --primary-800: #075985;
            --primary-900: #0c4a6e;
            
            --neutral-50: #f8fafc;
            --neutral-100: #f1f5f9;
            --neutral-200: #e2e8f0;
            --neutral-300: #cbd5e1;
            --neutral-400: #94a3b8;
            --neutral-500: #64748b;
            --neutral-600: #475569;
            --neutral-700: #334155;
            --neutral-800: #1e293b;
            --neutral-900: #0f172a;
            
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            
            --transition: all 0.3s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            min-height: 100vh;
            color: var(--neutral-800);
            line-height: 1.6;
        }

        .app-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .glass {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: var(--shadow-lg);
        }

        .app-header {
            background: white;
            border-radius: 24px 24px 0 0;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .logo {
            font-size: 3.5em;
            font-weight: 800;
            color: var(--primary-700);
            margin-bottom: 10px;
        }

        .logo i {
            color: var(--primary-500);
            margin-right: 10px;
        }

        .tagline {
            color: var(--neutral-600);
            font-size: 1.2em;
            font-weight: 400;
        }

        .status-bar {
            background: white;
            padding: 16px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--neutral-200);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--neutral-400);
            transition: var(--transition);
        }

        .status-dot.active {
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }

        .badge-success {
            background: #d1fae5;
            color: #065f46;
        }

        .badge-warning {
            background: #fed7aa;
            color: #92400e;
        }

        .main-content {
            background: white;
            padding: 30px;
            border-radius: 0 0 24px 24px;
        }

        .upload-section {
            background: linear-gradient(135deg, var(--primary-50), var(--primary-100));
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            border: 2px dashed var(--primary-300);
            transition: var(--transition);
        }

        .upload-section:hover {
            border-color: var(--primary-500);
            background: linear-gradient(135deg, var(--primary-100), var(--primary-200));
        }

        .file-input-wrapper {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }

        .file-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid var(--neutral-200);
            border-radius: 12px;
            background: white;
            font-size: 1em;
            transition: var(--transition);
        }

        .file-input:focus {
            outline: none;
            border-color: var(--primary-500);
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }

        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: var(--transition);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
            color: white;
            box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.2);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.3);
        }

        .btn-secondary {
            background: white;
            color: var(--primary-700);
            border: 2px solid var(--primary-200);
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--primary-50);
            border-color: var(--primary-400);
        }

        .search-section {
            background: var(--neutral-50);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
        }

        .search-box {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }

        .search-input {
            flex: 1;
            padding: 18px 24px;
            border: 2px solid var(--neutral-200);
            border-radius: 16px;
            font-size: 1.1em;
            transition: var(--transition);
            background: white;
        }

        .search-input:focus {
            border-color: var(--primary-500);
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1);
            outline: none;
        }

        .options-panel {
            background: white;
            border-radius: 16px;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            box-shadow: var(--shadow-sm);
        }

        .option-item {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .range-value {
            color: var(--primary-600);
            font-weight: 600;
            margin-left: 10px;
        }

        .results-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--neutral-200);
            padding-bottom: 10px;
        }

        .tab-btn {
            padding: 12px 24px;
            border: none;
            background: none;
            font-weight: 600;
            color: var(--neutral-600);
            cursor: pointer;
            transition: var(--transition);
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-btn:hover {
            color: var(--primary-600);
            background: var(--primary-50);
        }

        .tab-btn.active {
            color: var(--primary-700);
            background: var(--primary-100);
        }

        .result-card {
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: var(--shadow);
            transition: var(--transition);
            border: 1px solid var(--neutral-200);
            margin-bottom: 20px;
        }

        .card-header {
            background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
            color: white;
            padding: 20px;
        }

        .card-content {
            padding: 25px;
            max-height: 600px;
            overflow-y: auto;
        }

        .raw-chunk {
            background: var(--neutral-50);
            border-left: 4px solid var(--primary-500);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 0 12px 12px 0;
        }

        .raw-chunk pre {
            background: var(--neutral-800);
            color: #e5e7eb;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            margin-top: 10px;
        }

        .metadata-panel {
            background: linear-gradient(135deg, var(--neutral-50), white);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
            border: 1px solid var(--neutral-200);
        }

        .metadata-item {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--neutral-700);
            padding: 8px 16px;
            background: white;
            border-radius: 30px;
            box-shadow: var(--shadow-sm);
        }

        .metadata-item i {
            color: var(--primary-500);
        }

        .not-found-message {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 2px solid #ef4444;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(5px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading-content {
            background: white;
            padding: 40px;
            border-radius: 24px;
            text-align: center;
            box-shadow: var(--shadow-xl);
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid var(--neutral-200);
            border-top: 4px solid var(--primary-600);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .message {
            padding: 16px 24px;
            border-radius: 12px;
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .message-error { background: #fee2e2; color: #991b1b; border-left: 4px solid #ef4444; }
        .message-success { background: #d1fae5; color: #065f46; border-left: 4px solid #10b981; }
        .message-info { background: #dbeafe; color: #1e40af; border-left: 4px solid #3b82f6; }

        .progress-container {
            width: 100%;
            height: 8px;
            background: var(--neutral-200);
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-500), var(--primary-600));
            width: 0%;
            transition: width 0.3s ease;
        }

        .app-footer {
            text-align: center;
            margin-top: 30px;
            padding: 30px;
            background: white;
            border-radius: 20px;
            color: var(--neutral-600);
        }

        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: var(--neutral-50);
            border-radius: 16px;
            margin-bottom: 20px;
        }

        .chat-message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 12px;
        }

        .chat-message.query {
            background: #dbeafe;
            margin-left: 20%;
            border-bottom-right-radius: 4px;
        }

        .chat-message.answer {
            background: white;
            margin-right: 20%;
            border-bottom-left-radius: 4px;
            border: 1px solid var(--neutral-200);
        }

        .chat-message time {
            font-size: 0.8em;
            color: var(--neutral-500);
            display: block;
            margin-top: 5px;
        }

        .export-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: flex-end;
        }

        .btn-export {
            background: white;
            color: var(--neutral-700);
            border: 2px solid var(--neutral-200);
            padding: 10px 20px;
        }

        .btn-export:hover {
            border-color: var(--primary-500);
            color: var(--primary-700);
        }

        [data-tooltip] {
            position: relative;
            cursor: help;
        }

        [data-tooltip]:before {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 8px 12px;
            background: var(--neutral-800);
            color: white;
            border-radius: 6px;
            font-size: 0.85em;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 10;
        }

        [data-tooltip]:hover:before {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header glass">
            <div class="logo">
                <i class="fas fa-brain"></i> IntelliVault
            </div>
            <p class="tagline">Intelligent PDF Assistant with Semantic Search</p>
        </header>
        
        <!-- Status Bar -->
        <div class="status-bar glass">
            <div class="status-indicator">
                <span class="status-dot {{ 'active' if database else '' }}" id="statusDot"></span>
                <span class="status-text" id="statusText">
                    {% if database %}
                        <i class="fas fa-database"></i> Database Active ({{ db_count }} chunks)
                    {% else %}
                        <i class="fas fa-exclamation-triangle"></i> No Database - Upload a PDF
                    {% endif %}
                </span>
            </div>
            <div>
                {% if database %}
                    <span class="badge badge-success"><i class="fas fa-check-circle"></i> Ready</span>
                {% else %}
                    <span class="badge badge-warning"><i class="fas fa-clock"></i> Waiting</span>
                {% endif %}
            </div>
        </div>
        
        <!-- Main Content -->
        <main class="main-content glass">
            <!-- Upload Section -->
            <section class="upload-section" id="uploadSection">
                <div class="upload-header">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <h2>Upload Your Document</h2>
                </div>
                <div class="file-input-wrapper">
                    <input type="file" id="pdfFile" accept=".pdf" class="file-input" 
                           data-tooltip="Select a PDF file (max 50MB)">
                    <button class="btn btn-primary" onclick="uploadPDF()" id="uploadBtn">
                        <i class="fas fa-cog"></i> Process & Index
                    </button>
                </div>
                <div id="uploadStatus"></div>
                <div class="progress-container" id="uploadProgress" style="display: none;">
                    <div class="progress-bar" style="width: 50%;"></div>
                </div>
            </section>
            
            <!-- Search Section -->
            <section class="search-section">
                <div class="search-box">
                    <input type="text" id="query" class="search-input" 
                           placeholder="Ask anything about your PDF..."
                           {% if not database %}disabled{% endif %}
                           data-tooltip="Enter your question here">
                    <button class="btn btn-primary" onclick="search()" id="searchBtn"
                            {% if not database %}disabled{% endif %}>
                        <i class="fas fa-search"></i> Search
                    </button>
                </div>
                
                <div class="options-panel">
                    <div class="option-item">
                        <label><i class="fas fa-list"></i> Results:</label>
                        <select id="nResults">
                            <option value="1">1 section</option>
                            <option value="3">3 sections</option>
                            <option value="5" selected>5 sections</option>
                            <option value="7">7 sections</option>
                            <option value="10">10 sections</option>
                        </select>
                    </div>
                    <div class="option-item">
                        <label><i class="fas fa-random"></i> Diversity:</label>
                        <input type="range" id="diversity" min="0" max="0.5" step="0.1" value="0.3">
                        <span class="range-value" id="diversityValue">0.3</span>
                    </div>
                    <div class="option-item">
                        <label><i class="fas fa-eye"></i> Raw View:</label>
                        <input type="checkbox" id="showRaw" checked>
                    </div>
                </div>
            </section>
            
            <!-- Messages -->
            <div id="error" class="message message-error" style="display: none;"></div>
            <div id="success" class="message message-success" style="display: none;"></div>
            <div id="info" class="message message-info" style="display: none;"></div>
            
            <!-- Not Found Message -->
            <div id="notFound" class="not-found-message" style="display: none;">
                <i class="fas fa-search"></i>
                <h2>Information Not Found</h2>
                <p>The requested information could not be found in the current document.</p>
                <p style="margin-top: 15px;">💡 Try rephrasing your question or upload a different PDF.</p>
            </div>
            
            <!-- Results Tabs -->
            <div class="results-tabs" id="resultsTabs" style="display: none;">
                <button class="tab-btn active" onclick="switchTab('formatted')">
                    <i class="fas fa-magic"></i> Formatted View
                </button>
                <button class="tab-btn" onclick="switchTab('raw')">
                    <i class="fas fa-code"></i> Raw View
                </button>
                <button class="tab-btn" onclick="switchTab('chat')">
                    <i class="fas fa-history"></i> Chat History
                </button>
            </div>
            
            <!-- Results Container -->
            <div id="resultsContainer" style="display: none;">
                <!-- Metadata Panel -->
                <div class="metadata-panel" id="metadataPanel"></div>
                
                <!-- Formatted View -->
                <div id="formattedView" class="result-card">
                    <div class="card-header">
                        <h3><i class="fas fa-magic"></i> Formatted Output</h3>
                        <p>Beautifully formatted with intelligent structure detection</p>
                    </div>
                    <div class="card-content" id="formattedOutput"></div>
                </div>
                
                <!-- Raw View -->
                <div id="rawView" class="result-card" style="display: none;">
                    <div class="card-header">
                        <h3><i class="fas fa-code"></i> Raw Text Chunks</h3>
                        <p>Original extracted content</p>
                    </div>
                    <div class="card-content" id="rawOutput"></div>
                </div>
                
                <!-- Chat History View -->
                <div id="chatView" class="result-card" style="display: none;">
                    <div class="card-header">
                        <h3><i class="fas fa-history"></i> Chat History</h3>
                        <p>Your conversation with IntelliVault</p>
                    </div>
                    <div class="card-content">
                        <div id="chatHistory" class="chat-history"></div>
                        <div class="export-buttons">
                            <button class="btn btn-export" onclick="downloadChat()">
                                <i class="fas fa-download"></i> Download Chat
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </main>
        
        <!-- Footer -->
        <footer class="app-footer glass">
            <p>IntelliVault v2.0 - Intelligent PDF Assistant with Semantic Search</p>
        </footer>
    </div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <h3>Processing your request...</h3>
            <p style="color: var(--neutral-500); margin-top: 10px;">This may take a few seconds</p>
        </div>
    </div>
    
    <script>
        let chatMessages = [];
        
        document.getElementById('diversity').addEventListener('input', function(e) {
            document.getElementById('diversityValue').textContent = e.target.value;
        });

        const dropZone = document.getElementById('uploadSection');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                document.getElementById('pdfFile').files = files;
                showInfo(`Selected: ${files[0].name}`);
            } else {
                showError('Please drop a PDF file');
            }
        });

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('formattedView').style.display = tab === 'formatted' ? 'block' : 'none';
            document.getElementById('rawView').style.display = tab === 'raw' ? 'block' : 'none';
            document.getElementById('chatView').style.display = tab === 'chat' ? 'block' : 'none';
        }

        async function search() {
            const query = document.getElementById('query').value.trim();
            if (!query) { showError('Please enter a question'); return; }
            
            const nResults = document.getElementById('nResults').value;
            const diversity = document.getElementById('diversity').value;
            const showRaw = document.getElementById('showRaw').checked;
            
            document.getElementById('loadingOverlay').style.display = 'flex';
            document.getElementById('error').style.display = 'none';
            document.getElementById('success').style.display = 'none';
            document.getElementById('notFound').style.display = 'none';
            document.getElementById('searchBtn').disabled = true;
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query, n_results: parseInt(nResults), diversity: parseFloat(diversity) })
                });
                
                const data = await response.json();
                
                if (response.ok && data.success) {
                    chatMessages.push({ type: 'query', content: query, time: new Date().toLocaleTimeString() });
                    chatMessages.push({ type: 'answer', content: data.formatted_text, time: new Date().toLocaleTimeString() });
                    updateChatHistory();
                    showSuccess(`Found ${data.metadata.count} relevant sections`);
                    
                    if (showRaw) {
                        document.getElementById('rawOutput').innerHTML = data.raw_chunks.map((chunk, i) => 
                            `<div class="raw-chunk"><strong>Section ${i+1}</strong><pre>${escapeHtml(chunk)}</pre></div>`
                        ).join('');
                    } else {
                        document.getElementById('rawOutput').innerHTML = '<p style="color: var(--neutral-500); text-align: center;">Raw view is disabled.</p>';
                    }
                    
                    document.getElementById('formattedOutput').innerHTML = data.formatted_text;
                    
                    const metadataHtml = `
                        <div class="metadata-item"><i class="fas fa-clock"></i> ${data.metadata.timestamp}</div>
                        <div class="metadata-item"><i class="fas fa-search"></i> "${data.metadata.query}"</div>
                        <div class="metadata-item"><i class="fas fa-file-alt"></i> ${data.metadata.count} sections</div>
                        <div class="metadata-item"><i class="fas fa-chart-line"></i> Confidence: ${data.metadata.confidence || 'High'}</div>
                        <button class="btn btn-export" onclick="exportToPDF('${data.metadata.query}')"><i class="fas fa-file-pdf"></i> Export</button>
                    `;
                    document.getElementById('metadataPanel').innerHTML = metadataHtml;
                    
                    document.getElementById('resultsTabs').style.display = 'flex';
                    document.getElementById('resultsContainer').style.display = 'block';
                    // Default to formatted view active, raw hidden
                    document.getElementById('formattedView').style.display = 'block';
                    document.getElementById('rawView').style.display = 'none';
                    document.getElementById('chatView').style.display = 'none';
                    document.querySelector('.tab-btn.active').classList.remove('active');
                    document.querySelectorAll('.tab-btn')[0].classList.add('active');
                } else {
                    if (data.not_found) {
                        document.getElementById('notFound').style.display = 'block';
                    } else {
                        showError(data.error || 'Error searching');
                    }
                }
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                document.getElementById('loadingOverlay').style.display = 'none';
                document.getElementById('searchBtn').disabled = false;
            }
        }

        function updateChatHistory() {
            const chatHtml = chatMessages.map(msg => 
                `<div class="chat-message ${msg.type}"><strong>${msg.type === 'query' ? 'Q:' : 'A:'}</strong> ${msg.content}<time>${msg.time}</time></div>`
            ).join('');
            document.getElementById('chatHistory').innerHTML = chatHtml;
        }

        async function exportToPDF(query) {
            const content = document.getElementById('formattedOutput').innerHTML;
            try {
                showInfo('Generating PDF...');
                const response = await fetch('/export-pdf', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ content, query })
                });
                const data = await response.json();
                if (data.success) {
                    const link = document.createElement('a');
                    link.href = 'data:application/pdf;base64,' + data.pdf;
                    link.download = data.filename;
                    link.click();
                    showSuccess('PDF downloaded successfully!');
                } else {
                    showError(data.error || 'Failed to generate PDF');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            }
        }

        async function downloadChat() {
            if (chatMessages.length === 0) { showError('No chat history to download'); return; }
            try {
                showInfo('Generating chat PDF...');
                const response = await fetch('/download-chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ messages: chatMessages })
                });
                const data = await response.json();
                if (data.success) {
                    const link = document.createElement('a');
                    link.href = 'data:application/pdf;base64,' + data.pdf;
                    link.download = data.filename;
                    link.click();
                    showSuccess('Chat downloaded successfully!');
                } else {
                    showError(data.error || 'Failed to download chat');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            }
        }

        async function uploadPDF() {
            const fileInput = document.getElementById('pdfFile');
            const file = fileInput.files[0];
            if (!file) { showError('Please select a PDF file'); return; }
            if (file.size > 50 * 1024 * 1024) { showError('File size exceeds 50MB limit'); return; }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('loadingOverlay').style.display = 'flex';
            document.getElementById('uploadProgress').style.display = 'block';
            document.getElementById('uploadBtn').disabled = true;
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                if (response.ok && data.success) {
                    showSuccess(`✅ Success! Created ${data.chunks} chunks.`);
                    document.getElementById('statusText').innerHTML = `<i class="fas fa-database"></i> Database Active (${data.chunks} chunks)`;
                    document.getElementById('statusDot').className = 'status-dot active';
                    document.getElementById('query').disabled = false;
                    document.getElementById('searchBtn').disabled = false;
                    fileInput.value = '';
                } else {
                    showError(data.error || 'Upload failed');
                }
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                document.getElementById('loadingOverlay').style.display = 'none';
                document.getElementById('uploadProgress').style.display = 'none';
                document.getElementById('uploadBtn').disabled = false;
            }
        }

        function showError(m) { const d=document.getElementById('error'); d.innerHTML=`<i class="fas fa-exclamation-circle"></i> ${m}`; d.style.display='flex'; setTimeout(()=>d.style.display='none',5000); }
        function showSuccess(m) { const d=document.getElementById('success'); d.innerHTML=`<i class="fas fa-check-circle"></i> ${m}`; d.style.display='flex'; setTimeout(()=>d.style.display='none',5000); }
        function showInfo(m) { const d=document.getElementById('info'); d.innerHTML=`<i class="fas fa-info-circle"></i> ${m}`; d.style.display='flex'; setTimeout(()=>d.style.display='none',3000); }
        function escapeHtml(unsafe) { return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&#039;"); }
        
        document.getElementById('query').addEventListener('keypress', function(e) { if (e.key === 'Enter') search(); });
        
        async function checkDatabaseStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                if (data.database) {
                    document.getElementById('statusText').innerHTML = `<i class="fas fa-database"></i> Database Active (${data.chunks} chunks)`;
                    document.getElementById('statusDot').className = 'status-dot active';
                    document.getElementById('query').disabled = false;
                    document.getElementById('searchBtn').disabled = false;
                }
            } catch (error) { console.log('Status check failed'); }
        }
        setInterval(checkDatabaseStatus, 30000);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 IntelliVault - Intelligent PDF Assistant Starting...")
    print("="*70)
    print(f"📍 http://127.0.0.1:5000")
    print(f"📚 Database: {'Connected' if database else 'Not Connected'}")
    print(f"🔧 Features:")
    print(f"   • Semantic Search with Similarity Threshold ({SIMILARITY_THRESHOLD})")
    print(f"   • Professional Light UI")
    print(f"   • PDF Export (improved formatting)")
    print(f"   • Chat History Download")
    print(f"   • Smart Out-of-Document Detection")
    print("="*70 + "\n")
    app.run(debug=True, use_reloader=False, port=5000)