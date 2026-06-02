import os
import re
import chromadb
from typing import List, Dict, Optional, Tuple
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# UNIVERSAL PDF FORMATTER - Enhanced Version
# =====================================================
class UniversalPDFFormatter:
    """
    Universal formatter that works for ANY PDF content
    Detects and formats headings, lists, algorithms, tables, etc.
    """
    
    def __init__(self):
        # ===== HEADING DETECTION (Enhanced) =====
        self.heading_patterns = [
            # Chapter/Section/Unit patterns
            (r'^(CHAPTER|SECTION|UNIT|PART|MODULE|TOPIC|LESSON)\s+(\d+[.:]?\s*.*)$', 'h1', 1),
            (r'^(\d+)\.\s+([A-Z][A-Za-z\s]{2,})$', 'h1', 1),  # 1. Introduction
            (r'^(\d+\.\d+)\s+([A-Z][A-Za-z\s]{2,})$', 'h2', 2),  # 1.1 Background
            (r'^(\d+\.\d+\.\d+)\s+([A-Z][A-Za-z\s]{2,})$', 'h3', 3),  # 1.1.1 Details
            (r'^(\d+\.\d+\.\d+\.\d+)\s+([A-Z][A-Za-z\s]{2,})$', 'h4', 4),  # 1.1.1.1 Details
            
            # Roman numerals
            (r'^([IVX]+)\.\s+([A-Z][A-Za-z\s]{2,})$', 'h1', 1),  # I. Introduction
            (r'^([ivx]+)\.\s+([A-Z][A-Za-z\s]{2,})$', 'h2', 2),  # i. Introduction
            
            # Letter headings
            (r'^([A-Z])\.\s+([A-Z][A-Za-z\s]{2,})$', 'h2', 2),  # A. First Section
            (r'^([a-z])\)\s+([A-Z][A-Za-z\s]{2,})$', 'h3', 3),  # a) Subsection
            
            # ALL CAPS headings
            (r'^([A-Z][A-Z\s]{4,}[A-Z])$', 'h1', 1),  # INTRODUCTION TO AI
            (r'^([A-Z][A-Z\s]{2,}[A-Z]):$', 'h2', 2),  # INTRODUCTION:
            
            # Title Case headings
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})$', 'h2', 2),  # Artificial Intelligence
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?):$', 'h3', 3),  # Key Concepts:
            
            # Common section headings
            (r'^(Abstract|Introduction|Background|Methodology|Algorithm|Implementation|Results|Discussion|Conclusion|References|Appendix|Bibliography|Index)$', 'h1', 1),
            (r'^(Overview|Summary|Key Points|Important Notes|Definitions|Examples|Analysis|Proof|Theorem|Lemma|Corollary|Definition|Remark|Note)$', 'h2', 2),
        ]
        
        # ===== LIST DETECTION (Enhanced) =====
        self.list_patterns = [
            # Bullet points
            (r'^[•●○▪▶►◆◇⚫🔹🔸]\s+(.+)$', 'bullet'),
            (r'^[-–—]\s+(.+)$', 'bullet'),
            (r'^[*+]\s+(.+)$', 'bullet'),
            (r'^\s+(.+)$', 'bullet'),
            (r'^o\s+(.+)$', 'bullet'),
            (r'^-\s+(.+)$', 'bullet'),
            
            # Numbered lists
            (r'^(\d+)\.\s+(.+)$', 'numbered'),
            (r'^(\d+)\)\s+(.+)$', 'numbered'),
            (r'^(\d+)\s+(.+)$', 'numbered'),
            (r'^(\d+)[:.]\s+(.+)$', 'numbered'),
            
            # Lettered lists
            (r'^([a-z])\.\s+(.+)$', 'lettered'),
            (r'^([a-z])\)\s+(.+)$', 'lettered'),
            (r'^([A-Z])\.\s+(.+)$', 'lettered'),
            
            # Roman lists
            (r'^([ivx]+)\.\s+(.+)$', 'roman'),
            (r'^([ivx]+)\)\s+(.+)$', 'roman'),
            (r'^([IVX]+)\.\s+(.+)$', 'roman'),
        ]
        
        # ===== ALGORITHM PATTERNS =====
        self.algorithm_patterns = [
            (r'(Algorithm|Procedure|Function|Method|Pseudocode)\s+(\d*\.?\d*\s*:?\s*)([A-Za-z0-9_\s]+)', 'algorithm_header'),
            (r'^\s*(Step|Stage|Phase|Part)\s+(\d+)[:.)]\s*(.+)$', 'algorithm_step'),
            (r'^\s*(Input|Output|Require|Ensure|Return|Initialize|Let|Set|Print|Display):\s*(.+)$', 'algorithm_io'),
            (r'^\s*(if|then|else|for|while|repeat|until|return|break|continue|switch|case)\b', 'algorithm_keyword'),
        ]
        
        # ===== TABLE DETECTION =====
        self.table_patterns = [
            (r'^\|.+\|$', 'markdown_table'),
            (r'^\+[-+]+\+$', 'ascii_table'),
            (r'^\s*[0-9]+\s+[0-9]+\s+[0-9]+\s*$', 'numeric_table'),
        ]
        
        # ===== DEFINITION PATTERNS =====
        self.definition_patterns = [
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+[-–—:]\s+(.+?)(?=\.\s+[A-Z]|\n\n|$)', 'definition'),
            (r'“([^”]+)”\s+(?:is|means|refers to|defined as)\s+(.+?)(?=\.|$)', 'definition'),
            (r'\b([A-Z][a-z]+)\s+is\s+(.+?)(?=\.|$)', 'definition'),
            (r'\b([A-Z][a-z]+)\s+refers to\s+(.+?)(?=\.|$)', 'definition'),
        ]
        
        # ===== NOISE PATTERNS =====
        self.noise_patterns = [
            (r'--- Page \d+ ---', ''),
            (r'Page \d+ of \d+', ''),
            (r'^\s*\d+\s*$', ''),
            (r'\[PAGE \d+\]', ''),
            (r'', '•'),
            (r'�', ''),
        ]
    
    def clean_text(self, text: str) -> str:
        """Remove noise and clean text"""
        for pattern, replacement in self.noise_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        # Fix spacing issues
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s+\n', '\n\n', text)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def detect_heading(self, line: str) -> Tuple[Optional[int], str]:
        """
        Detect heading level (1,2,3,4) and return formatted heading
        """
        line = line.strip()
        if not line:
            return None, line
        
        for pattern, level_val, level_num in self.heading_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Format based on level
                if level_num == 1:
                    return 1, f'<h1 class="heading-1">{line}</h1>'
                elif level_num == 2:
                    return 2, f'<h2 class="heading-2">{line}</h2>'
                elif level_num == 3:
                    return 3, f'<h3 class="heading-3">{line}</h3>'
                elif level_num == 4:
                    return 4, f'<h4 class="heading-4">{line}</h4>'
        
        return None, line
    
    def format_lists(self, text: str) -> str:
        """Format all types of lists beautifully"""
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        list_type = None
        list_items = []
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                if in_list:
                    # End current list
                    formatted_lines.extend(self._render_list(list_type, list_items))
                    formatted_lines.append('')
                    in_list = False
                    list_type = None
                    list_items = []
                else:
                    formatted_lines.append('')
                i += 1
                continue
            
            # Check if line is a list item
            matched = False
            for pattern, list_style in self.list_patterns:
                match = re.match(pattern, line)
                if match:
                    if not in_list:
                        in_list = True
                        list_type = list_style
                        list_items = []
                    
                    # Extract the content (without the bullet/number)
                    if len(match.groups()) == 2:
                        marker, content = match.groups()
                    else:
                        content = match.group(1)
                    
                    list_items.append({
                        'marker': marker if len(match.groups()) == 2 else None,
                        'content': content,
                        'type': list_style,
                        'original': line
                    })
                    
                    matched = True
                    break
            
            if not matched:
                if in_list:
                    # End current list
                    formatted_lines.extend(self._render_list(list_type, list_items))
                    in_list = False
                    list_type = None
                    list_items = []
                formatted_lines.append(line)
            
            i += 1
        
        # Handle list at end of text
        if in_list:
            formatted_lines.extend(self._render_list(list_type, list_items))
        
        return '\n'.join(formatted_lines)
    
    def _render_list(self, list_type: str, items: List[Dict]) -> List[str]:
        """Render a list with proper HTML"""
        if not items:
            return []
        
        result = ['<div class="list-container">']
        
        if list_type == 'bullet':
            result.append('<ul class="bullet-list">')
            for item in items:
                result.append(f'  <li><span class="bullet-icon">•</span> {item["content"]}</li>')
            result.append('</ul>')
        
        elif list_type == 'numbered':
            result.append('<ol class="numbered-list">')
            for i, item in enumerate(items, 1):
                result.append(f'  <li><span class="list-number">{i}.</span> {item["content"]}</li>')
            result.append('</ol>')
        
        elif list_type == 'lettered':
            result.append('<ol class="lettered-list" type="a">')
            for i, item in enumerate(items):
                letter = chr(97 + i)  # a, b, c, ...
                result.append(f'  <li><span class="list-letter">{letter})</span> {item["content"]}</li>')
            result.append('</ol>')
        
        elif list_type == 'roman':
            result.append('<ol class="roman-list" type="i">')
            for i, item in enumerate(items, 1):
                roman = self._int_to_roman(i)
                result.append(f'  <li><span class="list-roman">{roman})</span> {item["content"]}</li>')
            result.append('</ol>')
        
        result.append('</div>')
        return result
    
    def _int_to_roman(self, num: int) -> str:
        """Convert integer to roman numeral"""
        val = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        roman = ''
        for n, r in val:
            while num >= n:
                roman += r
                num -= n
        return roman.lower()
    
    def format_algorithms(self, text: str) -> str:
        """Format algorithms beautifully"""
        lines = text.split('\n')
        formatted_lines = []
        in_algorithm = False
        algo_lines = []
        
        for line in lines:
            # Check for algorithm header
            if re.search(r'(Algorithm|Procedure|Function|Pseudocode)', line, re.IGNORECASE):
                if in_algorithm:
                    # End previous algorithm
                    formatted_lines.extend(self._render_algorithm(algo_lines))
                    algo_lines = []
                
                in_algorithm = True
                algo_lines.append(line)
            
            elif in_algorithm:
                if line.strip() == '' or re.search(r'end|return|}', line, re.IGNORECASE):
                    algo_lines.append(line)
                    formatted_lines.extend(self._render_algorithm(algo_lines))
                    in_algorithm = False
                    algo_lines = []
                else:
                    algo_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # Handle algorithm at end
        if in_algorithm and algo_lines:
            formatted_lines.extend(self._render_algorithm(algo_lines))
        
        return '\n'.join(formatted_lines)
    
    def _render_algorithm(self, lines: List[str]) -> List[str]:
        """Render an algorithm with proper styling"""
        if not lines:
            return []
        
        result = ['<div class="algorithm-box">']
        
        # First line is usually the header
        header = lines[0]
        result.append(f'<div class="algorithm-header">{header}</div>')
        result.append('<div class="algorithm-body">')
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Check for input/output
            if re.match(r'Input:|Output:|Initialize:|Return:', line, re.IGNORECASE):
                result.append(f'<div class="algorithm-io">{line}</div>')
            
            # Check for steps
            elif re.match(r'Step\s+\d+', line, re.IGNORECASE):
                result.append(f'<div class="algorithm-step">{line}</div>')
            
            # Check for keywords
            elif re.search(r'\b(if|then|else|for|while|return|break|continue|switch|case)\b', line, re.IGNORECASE):
                result.append(f'<div class="algorithm-keyword">{line}</div>')
            
            # Regular line
            else:
                result.append(f'<div class="algorithm-line">{line}</div>')
        
        result.append('</div></div>')
        return result
    
    def format_tables(self, text: str) -> str:
        """Format tables if detected"""
        lines = text.split('\n')
        formatted_lines = []
        in_table = False
        table_lines = []
        
        for line in lines:
            is_table = False
            for pattern, _ in self.table_patterns:
                if re.match(pattern, line):
                    is_table = True
                    break
            
            if is_table:
                in_table = True
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    formatted_lines.append(self._render_table(table_lines))
                    table_lines = []
                    in_table = False
                formatted_lines.append(line)
        
        if in_table and table_lines:
            formatted_lines.append(self._render_table(table_lines))
        
        return '\n'.join(formatted_lines)
    
    def _render_table(self, lines: List[str]) -> str:
        """Render a table with enhanced styling"""
        if not lines:
            return ''
        
        # Simple table rendering
        result = ['<div class="table-container"><table class="data-table">']
        
        # Detect if first row is header
        first_row = lines[0] if lines else ""
        is_header = not re.match(r'^[\+\-\|\=\s]+$', first_row)
        
        for i, line in enumerate(lines):
            # Skip separator lines
            if re.match(r'^[\+\-\|\=\s]+$', line):
                continue
            
            cells = re.split(r'\|', line)
            cells = [c.strip() for c in cells if c.strip()]
            
            if i == 0 and is_header:
                result.append('<thead><tr>')
                for cell in cells:
                    result.append(f'<th>{cell}</th>')
                result.append('</tr></thead><tbody>')
            else:
                if i == 1 and len(lines) > 1 and re.match(r'^[\+\-\|\=]+$', lines[1]):
                    continue
                result.append('<tr>')
                for cell in cells:
                    result.append(f'<td>{cell}</td>')
                result.append('</tr>')
        
        if '</tbody>' not in result[-1]:
            result.append('</tbody>')
        result.append('</table></div>')
        return '\n'.join(result)
    
    def format_definitions(self, text: str) -> str:
        """Format definitions with enhanced styling"""
        for pattern, def_type in self.definition_patterns:
            def replace_def(match):
                if len(match.groups()) == 2:
                    term, definition = match.groups()
                    return f'<div class="definition-box"><span class="term">{term}</span><span class="definition">{definition}</span></div>'
                return match.group(0)
            
            text = re.sub(pattern, replace_def, text, flags=re.IGNORECASE)
        
        return text
    
    def format_complete(self, text: str, query: str = "") -> str:
        """Complete formatting pipeline - Works for ALL PDFs"""
        
        # Clean the text first
        text = self.clean_text(text)
        
        # Apply formatting in sequence
        text = self.format_algorithms(text)
        text = self.format_tables(text)
        
        # Split into paragraphs for further processing
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Check if first line is a heading
            lines = para.split('\n')
            if len(lines) == 1:
                level, formatted = self.detect_heading(lines[0])
                if level:
                    formatted_paragraphs.append(formatted)
                    continue
            
            # Format lists within paragraph
            para = self.format_lists(para)
            para = self.format_definitions(para)
            
            # Wrap regular paragraphs
            if not para.startswith('<') and not para.startswith(' ') and len(para) > 0:
                # Highlight query terms if present
                if query:
                    para = self._highlight_query_terms(para, query)
                para = f'<p class="regular-text">{para}</p>'
            
            formatted_paragraphs.append(para)
        
        # Combine all content
        content = '\n\n'.join(formatted_paragraphs)
        
        # Create complete HTML document with comprehensive CSS
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Formatted Document - {query if query else "PDF Search"}</title>
    <style>
        /* Global Styles */
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
        }}
        
        /* Document Header */
        .document-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ transform: translateY(-20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        .document-header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header-meta {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 30px;
            font-size: 0.95em;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        /* Headings with animations */
        .heading-1 {{
            font-size: 2.2em;
            color: #2c3e50;
            margin: 30px 0 20px;
            padding-bottom: 10px;
            border-bottom: 4px solid #667eea;
            font-weight: 600;
            animation: fadeInLeft 0.5s ease-out;
        }}
        
        .heading-2 {{
            font-size: 1.8em;
            color: #34495e;
            margin: 25px 0 15px;
            padding-left: 15px;
            border-left: 5px solid #764ba2;
            font-weight: 500;
            animation: fadeInLeft 0.5s ease-out 0.1s both;
        }}
        
        .heading-3 {{
            font-size: 1.4em;
            color: #5d6d7e;
            margin: 20px 0 10px;
            padding-left: 25px;
            font-weight: 500;
            position: relative;
            animation: fadeInLeft 0.5s ease-out 0.2s both;
        }}
        
        .heading-4 {{
            font-size: 1.2em;
            color: #7f8c8d;
            margin: 15px 0 10px;
            padding-left: 35px;
            font-weight: 500;
            font-style: italic;
            animation: fadeInLeft 0.5s ease-out 0.3s both;
        }}
        
        .heading-3::before, .heading-4::before {{
            content: '►';
            color: #667eea;
            position: absolute;
            left: 5px;
            font-size: 0.9em;
        }}
        
        @keyframes fadeInLeft {{
            from {{ transform: translateX(-20px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        /* Lists with enhanced styling */
        .list-container {{
            margin: 15px 0;
            animation: fadeIn 0.5s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .bullet-list, .numbered-list, .lettered-list, .roman-list {{
            padding: 15px 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .bullet-list:hover, .numbered-list:hover, .lettered-list:hover, .roman-list:hover {{
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
        }}
        
        .bullet-list li, .numbered-list li, .lettered-list li, .roman-list li {{
            margin: 8px 0;
            line-height: 1.6;
            transition: color 0.3s ease;
        }}
        
        .bullet-list li:hover, .numbered-list li:hover, .lettered-list li:hover, .roman-list li:hover {{
            color: #667eea;
        }}
        
        .bullet-icon {{
            color: #667eea;
            font-weight: bold;
            margin-right: 10px;
            display: inline-block;
            width: 20px;
        }}
        
        .list-number, .list-letter, .list-roman {{
            color: #764ba2;
            font-weight: bold;
            margin-right: 10px;
            display: inline-block;
            min-width: 30px;
        }}
        
        /* Algorithm Styling with glow effect */
        .algorithm-box {{
            background: #1e1e2f;
            border-radius: 15px;
            margin: 30px 0;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 1px solid #2d2d4a;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: glowPulse 3s infinite;
        }}
        
        @keyframes glowPulse {{
            0% {{ box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }}
            50% {{ box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5); }}
            100% {{ box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }}
        }}
        
        .algorithm-box:hover {{
            transform: scale(1.02);
        }}
        
        .algorithm-header {{
            background: linear-gradient(135deg, #2d2d4a, #1a1a2e);
            color: #ffd700;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 1.2em;
            border-bottom: 2px solid #4a4a8a;
            font-family: 'Courier New', monospace;
        }}
        
        .algorithm-body {{
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
        }}
        
        .algorithm-step {{
            color: #4a90e2;
            padding: 5px 10px;
            margin: 5px 0;
            border-left: 3px solid #4a90e2;
            transition: background 0.3s ease;
        }}
        
        .algorithm-step:hover {{
            background: #252538;
        }}
        
        .algorithm-io {{
            color: #00b894;
            padding: 5px 10px;
            margin: 5px 0;
            background: #252538;
            border-radius: 5px;
            border-left: 3px solid #00b894;
        }}
        
        .algorithm-keyword {{
            color: #ffd700;
            padding: 5px 10px;
            margin: 5px 0;
            font-weight: bold;
            background: rgba(255, 215, 0, 0.1);
            border-radius: 5px;
        }}
        
        .algorithm-line {{
            padding: 3px 10px;
            margin: 3px 0;
            color: #b0b0b0;
        }}
        
        /* Tables with hover effects */
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
            animation: fadeIn 0.5s ease-out;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .data-table:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            position: relative;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            transition: background 0.3s ease;
        }}
        
        tr:nth-child(even) {{
            background: #f8fafc;
        }}
        
        tr:hover td {{
            background: #f1f5f9;
        }}
        
        /* Definition Boxes with flip animation */
        .definition-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideInRight 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        @keyframes slideInRight {{
            from {{ transform: translateX(20px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        .definition-box::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            transform: translateX(-100%);
            transition: transform 0.5s ease;
        }}
        
        .definition-box:hover::before {{
            transform: translateX(0);
        }}
        
        .definition-box:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
        }}
        
        .term {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
            display: block;
            margin-bottom: 5px;
            position: relative;
            z-index: 1;
        }}
        
        .definition {{
            color: #4a5568;
            line-height: 1.6;
            padding-left: 15px;
            border-left: 2px solid #cbd5e0;
            position: relative;
            z-index: 1;
        }}
        
        /* Regular Text with highlight */
        .regular-text {{
            line-height: 1.8;
            margin: 15px 0;
            color: #2d3748;
            text-align: justify;
            animation: fadeIn 0.5s ease-out;
            transition: color 0.3s ease;
        }}
        
        .regular-text:hover {{
            color: #1a202c;
        }}
        
        .highlight {{
            background: linear-gradient(120deg, #fff3bf 0%, #ffe69b 100%);
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: 500;
            animation: pulse 2s infinite;
            display: inline-block;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        /* Footer */
        .document-footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #718096;
            border-top: 2px solid #e2e8f0;
            font-style: italic;
            animation: fadeIn 1s ease-out;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .heading-1 {{ font-size: 1.8em; }}
            .heading-2 {{ font-size: 1.5em; }}
            .heading-3 {{ font-size: 1.2em; }}
            .heading-4 {{ font-size: 1.1em; }}
            .document-header {{ padding: 20px; }}
            .document-header h1 {{ font-size: 1.8em; }}
        }}
        
        /* Code blocks */
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #e53e3e;
            transition: all 0.3s ease;
        }}
        
        code:hover {{
            background: #e2e8f0;
            color: #c53030;
        }}
        
        pre {{
            background: #1e1e2f;
            color: #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            border-left: 4px solid #667eea;
            transition: transform 0.3s ease;
        }}
        
        pre:hover {{
            transform: scale(1.01);
        }}
        
        /* Blockquotes */
        blockquote {{
            border-left: 4px solid #667eea;
            padding: 10px 20px;
            margin: 15px 0;
            background: #f8fafc;
            font-style: italic;
            border-radius: 0 8px 8px 0;
            transition: all 0.3s ease;
        }}
        
        blockquote:hover {{
            background: #f1f5f9;
            border-left-width: 6px;
        }}
        
        /* Horizontal rule */
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            margin: 30px 0;
            animation: expandWidth 1s ease-out;
        }}
        
        @keyframes expandWidth {{
            from {{ width: 0; opacity: 0; }}
            to {{ width: 100%; opacity: 1; }}
        }}
        
        /* Tooltips for definitions */
        [data-tooltip] {{
            position: relative;
            cursor: help;
        }}
        
        [data-tooltip]:before {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 5px 10px;
            background: #333;
            color: white;
            border-radius: 5px;
            font-size: 0.9em;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            z-index: 10;
        }}
        
        [data-tooltip]:hover:before {{
            opacity: 1;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .document-header {{
                background: #667eea;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            
            .algorithm-box {{
                break-inside: avoid;
            }}
            
            .definition-box {{
                break-inside: avoid;
            }}
            
            .data-table {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="document-header">
        <h1>📄 {query if query else "Formatted Document"}</h1>
        <div class="header-meta">
            <span class="meta-item">🔍 Query: {query}</span>
            <span class="meta-item">📅 {timestamp}</span>
            <span class="meta-item">✨ Universal Formatter</span>
            <span class="meta-item">📊 {len(re.findall(r'<[^>]+>', content))} elements</span>
        </div>
    </div>
    
    <div class="content">
        {content}
    </div>
    
    <div class="document-footer">
        <p>✨ Formatted with Universal PDF Formattr • Works with ALL PDFs</p>
        <p style="font-size: 0.9em; margin-top: 10px;">
            <i class="fas fa-check-circle"></i> Headings • <i class="fas fa-list"></i> Lists • 
            <i class="fas fa-code"></i> Algorithms • <i class="fas fa-table"></i> Tables • 
            <i class="fas fa-bookmark"></i> Definitions
        </p>
    </div>
</body>
</html>'''
        
        return html
    
    def _highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in the text"""
        if not query:
            return text
        
        # Split query into words and escape for regex
        query_words = re.findall(r'\w+', query.lower())
        for word in query_words:
            if len(word) > 2:  # Only highlight words longer than 2 characters
                pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
                text = pattern.sub(r'<span class="highlight">\1</span>', text)
        
        return text


# =====================================================
# Enhanced Embedding and Search Functions
# =====================================================

class EnhancedEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_fitted = False
    
    def __call__(self, input: Documents) -> Embeddings:
        # Get sentence embeddings
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for better similarity
        )
        return embeddings.tolist()
    
    def fit_tfidf(self, documents: List[str]):
        """Fit TF-IDF vectorizer on documents"""
        if len(documents) > 0:
            self.tfidf_vectorizer.fit(documents)
            self.is_fitted = True
    
    def get_hybrid_scores(self, query: str, documents: List[str], semantic_embeddings: List[List[float]], 
                          semantic_weight: float = 0.7, tfidf_weight: float = 0.3) -> List[float]:
        """
        Combine semantic and TF-IDF scores for better retrieval
        """
        # Semantic similarity (cosine)
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        semantic_scores = [
            np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in semantic_embeddings
        ]
        
        # TF-IDF similarity if fitted
        if self.is_fitted and len(documents) > 0:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                docs_tfidf = self.tfidf_vectorizer.transform(documents)
                tfidf_scores = cosine_similarity(query_tfidf, docs_tfidf).flatten()
            except:
                tfidf_scores = np.zeros(len(documents))
        else:
            tfidf_scores = np.zeros(len(documents))
        
        # Combine scores
        hybrid_scores = semantic_weight * np.array(semantic_scores) + tfidf_weight * tfidf_scores
        
        return hybrid_scores.tolist()


# Initialize embedding function
embedding_model = EnhancedEmbeddingFunction()


def load_pdf(file_path: str) -> str:
    """Load PDF and extract text with better formatting"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                # Add page marker for reference
                text += f"\n\n[PAGE {page_num}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""


def split_text_smart(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Smart text splitting that respects paragraph and sentence boundaries
    """
    # Remove page markers for chunking but keep for reference
    clean_text = re.sub(r'\[PAGE \d+\]', '', text)
    
    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', clean_text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_words = len(para.split())
        
        # If paragraph is too large, split into sentences
        if para_words > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_words = len(sent.split())
                if current_size + sent_words <= chunk_size:
                    current_chunk.append(sent)
                    current_size += sent_words
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_size = sent_words
        else:
            if current_size + para_words <= chunk_size:
                current_chunk.append(para)
                current_size += para_words
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_size = para_words
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Create overlapping chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            overlapped_chunks.append(chunks[i])
            if i < len(chunks) - 1:
                # Create overlap with next chunk
                words1 = chunks[i].split()
                words2 = chunks[i + 1].split()
                overlap_words = words1[-overlap:] + words2[:overlap]
                overlapped_chunks.append(' '.join(overlap_words))
        chunks = overlapped_chunks
    
    return chunks


def create_chroma_db(documents: List[str], path: str, name: str):
    """Create ChromaDB collection with enhanced embeddings"""
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path=path)

    try:
        client.delete_collection(name)
    except:
        pass

    collection = client.create_collection(
        name=name,
        embedding_function=embedding_model
    )

    # Fit TF-IDF on documents
    embedding_model.fit_tfidf(documents)

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = [f"doc_{i + j}" for j in range(len(batch_docs))]
        
        # Add metadata for each chunk
        metadatas = [{"chunk_index": i + j, "source": "pdf"} for j in range(len(batch_docs))]
        
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=metadatas
        )

    return collection


def load_chroma_collection(path: str, name: str):
    """Load existing ChromaDB collection"""
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(
        name=name,
        embedding_function=embedding_model
    )


def diversity_reranking(documents, embeddings, similarities, n_results, diversity_factor):
    """
    Rerank results to promote diversity
    """
    if len(documents) <= n_results:
        return list(range(len(documents)))
    
    selected = [0]  # Start with the most similar
    candidates = list(range(1, len(documents)))
    
    while len(selected) < n_results and candidates:
        best_candidate = None
        best_score = -float('inf')
        
        for i in candidates:
            if i in selected:
                continue
            
            # Relevance score
            relevance = similarities[i]
            
            # Diversity score (min similarity with selected)
            diversity = min(
                1 - np.dot(embeddings[i], embeddings[j])
                for j in selected
            )
            
            # Combined score
            score = (1 - diversity_factor) * relevance + diversity_factor * diversity
            
            if score > best_score:
                best_score = score
                best_candidate = i
        
        if best_candidate is not None:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
    
    return selected


def retrieve_similar_text_enhanced(query: str, collection, n_results: int = 5, 
                                   diversity_factor: float = 0.3,
                                   similarity_threshold: float = 0.2) -> List[str]:
    """
    Enhanced retrieval with hybrid (semantic + TF‑IDF) scoring and diversity reranking.
    Returns chunks with hybrid score >= threshold. If none, returns top n_results anyway.
    """
    if collection.count() == 0:
        return []
    
    try:
        # Get more results than needed for diversity
        initial_results = min(n_results * 3, collection.count())
        
        results = collection.query(
            query_texts=[query],
            n_results=initial_results,
            include=["documents", "distances", "metadatas", "embeddings"]
        )

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        embeddings = results["embeddings"][0] if "embeddings" in results else []
        
        if not documents:
            return []
        
        # Convert distances to similarity scores (distance = 1 - similarity for cosine)
        similarities = [1 - d for d in distances] if distances else []
        
        # Compute hybrid scores (semantic + TF‑IDF)
        hybrid_scores = embedding_model.get_hybrid_scores(query, documents, embeddings)
        
        # ---- STRICT THRESHOLD LOGIC FOR OUT-OF-DOCUMENT DETECTION ----
        # First, try to filter by threshold
        filtered = [(doc, score, emb) for doc, score, emb in zip(documents, hybrid_scores, embeddings) 
                    if score >= similarity_threshold]
        
        # If nothing passes the threshold, return empty list (reject out-of-document questions)
        if not filtered:
            print(f"⚠️ No chunks above threshold {similarity_threshold}. Question appears to be outside document scope.")
            return []
        
        # Unpack filtered results
        documents, hybrid_scores, embeddings = zip(*filtered)
        documents = list(documents)
        hybrid_scores = list(hybrid_scores)
        embeddings = list(embeddings)
        
        # Apply diversity reranking using hybrid scores as relevance
        if len(documents) > n_results and diversity_factor > 0 and embeddings:
            selected_indices = diversity_reranking(
                documents, embeddings, hybrid_scores, n_results, diversity_factor
            )
            documents = [documents[i] for i in selected_indices]
        else:
            documents = documents[:n_results]
        
        return documents
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_retrieved_text_professionally(retrieved_chunks: List[str], query: str = "") -> Dict:
    """Format retrieved text with universal formatting"""
    if not retrieved_chunks:
        return {
            "success": False,
            "raw_chunks": [],
            "formatted_text": "<div class='error-message'>No content to format.</div>",
            "metadata": {"count": 0}
        }
    
    formatter = UniversalPDFFormatter()
    combined_raw = "\n\n".join(retrieved_chunks)
    formatted = formatter.format_complete(combined_raw, query)
    
    return {
        "success": True,
        "raw_chunks": retrieved_chunks,
        "formatted_text": formatted,
        "metadata": {
            "count": len(retrieved_chunks),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query
        }
    }


def process_pdf_and_create_db(pdf_path: str, db_path: str = "./chroma_db", collection_name: str = "pdf_docs"):
    """Process PDF and create database"""
    print(f"📄 Processing PDF: {pdf_path}")
    raw_text = load_pdf(pdf_path)
    if not raw_text:
        print("❌ Failed to load PDF")
        return None
    
    chunks = split_text_smart(raw_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    collection = create_chroma_db(chunks, db_path, collection_name)
    return collection


# Export all functions
__all__ = [
    'UniversalPDFFormatter',
    'load_pdf',
    'split_text_smart',
    'create_chroma_db',
    'load_chroma_collection',
    'retrieve_similar_text_enhanced',
    'format_retrieved_text_professionally',
    'process_pdf_and_create_db',
    'EnhancedEmbeddingFunction'
]