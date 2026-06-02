# IntelliVault - Intelligent PDF Assistant

An intelligent PDF assistant with semantic search capabilities powered by AI. Upload your PDF documents and ask questions to get relevant, formatted answers.

## 🚀 Features

- **Semantic Search with Similarity Threshold (0.2)** - Smart content detection
- **Professional Light UI** - Modern, clean interface with animations
- **PDF Export** - Export search results with improved formatting
- **Chat History Download** - Save your conversation as PDF
- **Smart Out-of-Document Detection** - Only answers questions within PDF scope
- **Hybrid Scoring** - Combines semantic similarity (70%) + TF-IDF (30%)
- **Multiple Result Formats** - Formatted view, raw view, and chat history

## 📋 Requirements

- Python 3.10+
- Flask
- ChromaDB 1.5.9+
- sentence-transformers
- Google Generative AI
- ReportLab
- WeasyPrint
- PyPDF

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd IntelliVault-main
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

3. Run the application:
```bash
python app7.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 💡 How to Use

1. **Upload PDF**: Click "Choose file" and upload your PDF document
2. **Process**: Click "Process & Index" to create the vector database
3. **Search**: Type your question in the search box
4. **Adjust Settings**:
   - Results: Choose number of sections (1-10)
   - Diversity: Adjust result diversity (0-0.5)
   - Raw View: Toggle to see original chunks
5. **Export**: Download results as formatted PDF
6. **Chat History**: View and download your conversation

## ⚙️ Configuration

### Similarity Threshold
The application uses a similarity threshold of **0.2 (20%)** to detect out-of-document questions:
- Questions matching content in PDF → Returns relevant answers ✅
- Questions not in PDF → Shows "Information Not Found" ❌

You can adjust this in `app7.py`:
```python
SIMILARITY_THRESHOLD = 0.2  # Adjust between 0.0 - 1.0
```

## 🏗️ Architecture

- **Backend**: Flask web framework
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **PDF Processing**: PyPDF with smart text extraction
- **Formatting**: Universal PDF formatter with heading/list detection
- **Scoring**: Hybrid semantic + TF-IDF scoring

## 📁 Project Structure

```
IntelliVault-main/
├── app7.py              # Main Flask application
├── model7.py            # PDF processing and retrieval logic
├── evalution.py         # Evaluation module (stub)
├── requirement.txt      # Python dependencies
├── uploads/             # Temporary PDF storage
├── chroma_db/           # Vector database storage
└── exports/             # Exported PDF files
```

## 🎯 Key Components

### model7.py
- `UniversalPDFFormatter`: Formats any PDF content with headings, lists, algorithms
- `EnhancedEmbeddingFunction`: Hybrid scoring with semantic + TF-IDF
- `retrieve_similar_text_enhanced`: Smart retrieval with threshold filtering
- `format_retrieved_text_professionally`: Beautiful HTML formatting

### app7.py
- `/upload`: Process and index PDF documents
- `/search`: Semantic search with threshold detection
- `/export-pdf`: Export formatted results
- `/download-chat`: Download conversation history
- `/evaluate`: Evaluation endpoint (placeholder)

## 🔧 Troubleshooting

**Issue**: "Information Not Found" for relevant questions
- **Solution**: Lower the `SIMILARITY_THRESHOLD` in app7.py

**Issue**: ChromaDB compatibility errors
- **Solution**: Upgrade to ChromaDB 1.5.9+ for Python 3.14 support

**Issue**: Model download slow
- **Solution**: First run downloads ~90MB model, subsequent runs are fast

## 📝 License

This project is open source and available under the MIT License.

## 👥 Author

Bhargavi (desaboyinabhargavi@gmail.com)

## 🙏 Acknowledgments

- sentence-transformers for embedding models
- ChromaDB for vector database
- Flask for web framework
- ReportLab for PDF generation
