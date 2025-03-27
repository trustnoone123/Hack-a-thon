# Automotive Drawing Analysis System

## Project Overview
This project is an advanced technical drawing analysis system designed specifically for automotive engineering documents. It leverages machine learning, optical character recognition (OCR), and AI-powered analysis to extract and process critical information from technical drawings.

## Key Features
- 📄 **Multi-Image Processing**: Upload multiple technical drawings simultaneously
- 🔍 **Intelligent Text Extraction**: Uses Tesseract OCR to extract drawing metadata
- 🤖 **AI-Powered Analysis**: Leverages GPT-4o-mini for technical drawing insights
- 🎯 **Symbol Detection**: Roboflow AI for identifying technical symbols
- 📊 **Data Management**: Exports extracted data to Excel and Neo4j graph database

## Technology Stack
- **Languages**: Python
- **Libraries/Frameworks**:
  - OpenCV (Image Processing)
  - Pytesseract (OCR)
  - Roboflow (Symbol Detection)
  - OpenAI GPT-4o-mini (AI Analysis)
  - Neo4j (Graph Database)
  - Streamlit (Web Interface)
  - Pandas (Data Management)

## Prerequisites
- Python 3.8+
- Tesseract OCR
- API Keys:
  - OpenAI
  - Roboflow
- Neo4j Database

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/automotive-drawing-analysis.git
cd automotive-drawing-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
- Update `OPENAI_API_KEY` in `mainExtractionOCR.py`
- Configure Roboflow API key
- Set up Neo4j database credentials

### 4. Set Tesseract Path
Update the Tesseract executable path in both Python scripts:
```python
pytesseract.pytesseract.tesseract_cmd = 'PATH_TO_TESSERACT'
```

## How It Works

### Drawing Processing Workflow
1. **Image Upload**: User uploads technical drawing images
2. **Preprocessing**: 
   - Convert to grayscale
   - Remove noise
   - Isolate text regions
3. **OCR Extraction**: 
   - Extract metadata like drawing number, title, etc.
4. **Symbol Detection**: 
   - Identify and classify technical symbols
5. **AI Analysis**: 
   - Generate detailed drawing insights
6. **Data Export**: 
   - Excel spreadsheet
   - Neo4j graph database

### Key Components

#### 1. `drawingNum.py`
- Specialized OCR function for extracting specific drawing metadata
- Advanced image preprocessing techniques
- Configurable keyword matching

#### 2. `mainExtractionOCR.py`
- Main Streamlit application
- Integrates multiple AI and data processing services
- Handles image upload, processing, and visualization

## Configuration Options
- Adjust OCR confidence thresholds
- Customize symbol detection parameters
- Modify AI analysis prompts

## Security Considerations
- Securely manage API keys
- Use environment variables
- Implement proper error handling

## Limitations
- Requires clear, high-quality technical drawings
- Performance depends on drawing complexity
- Symbol detection accuracy varies

## Future Improvements
- Support more file formats
- Enhanced AI analysis
- Real-time collaboration features

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify License - e.g., MIT, Apache 2.0]

## Contact
- Project Developers: Jhanani, Balaji, Shakthi, Swetha
- Contact: [Your Contact Information]
```

