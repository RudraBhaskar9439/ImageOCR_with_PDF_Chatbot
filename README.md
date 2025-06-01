
# ImageOCR and PDF Knowledge Base with Gemini Vision
A Python application that combines OCR (Optical Character Recognition) capabilities with PDF processing using Google's Gemini Vision API and semantic search.

# 🎯 Features
📸 Image OCR using Gemini Vision API

📄 PDF text extraction and processing


🔍 Semantic search with sentence transformers


💭 Conversation memory for context-aware responses

📊 Similarity-based content retrieval

🤖 Integration with Google's Gemini AI

# 📋 Prerequisites

Python 3.8+

Google Gemini API key

PDF documents and/or images to process

# 🛠️ Installation
## 1. Clone the repository
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
## 2. Install required packages
```python
pip3 install google-generativeai sentence-transformers PyMuPDF langchain python-dotenv numpy Pillow
```
## 3. Set up environment variables
```python
cp .env.example .env
# Edit .env with your API key
```
# 📁 Project Structure
```python
.
├── imageOCR.py           # Main application file
├── Folder_with_pdfs/     # Directory for PDFs and images
├── .env                  # Environment variables
├── .gitignore           # Git ignore file
└── README.md            # Documentation
```

# 💻 Usage
1. Place your files in the Folder_with_pdfs directory:

Supported image formats: .png, .jpg, .jpeg, .tiff, .bmp

Supported document format: .pdf

2. Run the script:

```python
python3 imageOCR.py
```

3. Interact with your documents:

```python
Enter your question (or 'quit' to exit): What text is visible in image1.jpg?
```

# ⚙️ Configuration

Customize these parameters in the code:
```python
chunk_size = 1000          # Size of text chunks
chunk_overlap = 200        # Overlap between chunks
max_history = 5           # Conversation memory size
top_k = 3                # Number of relevant chunks
```

# 🔒 Security
Never commit your .env file

Keep your API keys secure

Use environment variables for sensitive data

# 🤝 Contributing

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

# MIT License

Copyright (c) 2024 [Rudra Bhaskar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Security Note
⚠️ Never commit your .env file or expose your API keys.

Author
[Rudra Bhaskar]