

# ---this is mainExtractionOCR.py-----------------------------------------------------------
import base64
import openai
import streamlit as st
import cv2
import os
import pytesseract
import numpy as np
from openpyxl import Workbook, load_workbook
import json
import importlib.util
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')
# Import the GetString function from drawingNum.py
spec = importlib.util.spec_from_file_location("drawingNum", "drawingNum.py")
drawingNum_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawingNum_module)
GetString = drawingNum_module.GetString

# Set Tesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def encode_image(image_path):
    """
    Encode an image to base64
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_drawing_with_gpt(extracted_image_path, json_data):
    """
    Analyze the extracted drawing using OpenAI's GPT-4o-mini
    
    Args:
        extracted_image_path (str): Path to the extracted image
        json_data (dict): Extracted metadata from the drawing
    
    Returns:
        str: AI-generated description of the drawing
    """
    try:
        # Encode the image
        base64_image = encode_image(extracted_image_path)
        
        # Prepare the prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert automotive engineer and technical drawing analyst. "
                               "Describe the technical drawing in detail from an automotive perspective, "
                               "focusing on dimensional details, component specifications, and design insights."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": f"Analyze this technical drawing with the following metadata: {json.dumps(json_data)}"
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error analyzing drawing: {str(e)}"


# def analyze_drawing_with_llm(image_path, json_data):
#     """
#     Analyze the extracted drawing using OpenAI's GPT-4o-mini
    
#     Args:
#     image_path (str): Path to the extracted drawing image
#     json_data (dict): Extracted metadata from the drawing
    
#     Returns:
#     str: LLM analysis of the drawing
#     """
#     # Read the image and convert to base64
#     with open(image_path, "rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
#     # Prepare the prompt with both image and metadata
#     prompt = f"""Analyze this technical drawing from an automotive perspective. 
#     Consider the drawing's details, potential component or system representation, 
#     and provide a comprehensive technical description.

#     Extracted Metadata:
#     {json.dumps(json_data, indent=2)}

#     Please provide:
#     1. Detailed description of the drawing's content
#     2. Potential automotive application or system
#     3. Any dimensional or technical insights visible
#     4. Potential manufacturing or design considerations"""

#     try:
#         response = openai.responses.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "input": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{base64_image}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             max_tokens=1000
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error in LLM analysis: {str(e)}"

def process_images(uploaded_files):
    # Ensure output directories exist
    os.makedirs('extracted', exist_ok=True)
    os.makedirs('json_format', exist_ok=True)

    # Create a new workbook
    wb = Workbook()
    
    # Store JSON data
    json_data = {}
    extracted_paths = []

    for image_file in uploaded_files:
        # Read the image
        img_array = np.frombuffer(image_file.read(), np.uint8)
        init_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        # Generate a unique filename
        filename = image_file.name
        image_number = filename.split('.')[0]

        [init_row, init_col] = init_img.shape
        
        # --- Cropping + border image --- #
        img = init_img[12:init_row-15, 12:init_col-12]
        [nrow, ncol] = img.shape
        
        # --- Isolating vertical & horizontal lines --- #
        ret, bin_img= cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ncol//150))
        eroded_verti = cv2.erode(bin_img, horiz_kernel, iterations = 5)
        vertical_lines = cv2.dilate(eroded_verti, horiz_kernel, iterations = 5)
        
        verti_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nrow//150, 1))
        eroded_hori = cv2.erode(bin_img, verti_kernel, iterations=5)
        horizontal_lines = cv2.dilate(eroded_hori, verti_kernel, iterations = 5)
        
        combined_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
        
        # --- Drawing remove --- #
        rect_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        drawingMask = cv2.erode(combined_lines, rect_kernel3, iterations = 2)
        drawingMask = cv2.dilate(drawingMask, rect_kernel3, iterations = 50)
        table_lines = drawingMask + np.bitwise_not(combined_lines)
        
        # --- Removing arrow lines --- #
        table_lines_dil = cv2.dilate(np.bitwise_not(table_lines), rect_kernel3, iterations = 5)
        
        contours, hierarchy = cv2.findContours(table_lines_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse = False)
        
        # --- Filling remaining drawing contours w/ white --- #
        table_bgr = cv2.cvtColor(table_lines, cv2.COLOR_GRAY2BGR)
        
        for i in range(0, len(sorted_contours)):
            cntr = sorted_contours[i]
            x, y, w, h = cv2.boundingRect(cntr)
            if (w < 30 or h < 30):
                cv2.drawContours(table_bgr, sorted_contours, i, (255, 255, 255), thickness=-1)
        
        table_only = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
        _, table_only = cv2.threshold(table_only, 150, 255, cv2.THRESH_BINARY)
                
        # --- Isolating table cells --- #
        table_only_copy = cv2.copyMakeBorder(table_only, 5, 5, 5, 5, cv2.BORDER_CONSTANT, 0)
        table_lines_dil2 = cv2.dilate(np.bitwise_not(table_only_copy), rect_kernel3, iterations = 1)
        cell_cntr, hierarchy = cv2.findContours(table_lines_dil2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- Creating mask for tables + obtaining coordinates for useful cells --- #
        table_bgr2 = cv2.cvtColor(table_only, cv2.COLOR_GRAY2BGR)
        
        keywords = ["DRAWING NUMBER", "DRAWING NO", "DRAWN BY", "DRAWN", "CHECKED BY", "CHECKED", "TITLE", "DRAWING TITLE", "APPPROVED BY", "APPROVED", "CONTRACTOR", "COMPANY", "UNIT", "STATUS", "PAGE", "PROJECT NO", "PROJECT NUM", "LANG", "CAD NO", "FONT", "FONT STYLE", "AMENDMENTS"]
        useful_cells = []
        
        for c in cell_cntr:
            coordinates = cv2.boundingRect(c)
            x, y, w, h = coordinates
            rect_area = w * h
            if (rect_area < ((nrow//4) * (ncol//4)) and h < 400):
                cell = img[y:y+h, x:x+w]
                string = (pytesseract.image_to_string(cell, config ='--psm 6')).strip()
                string_list = string.splitlines()
                for k in keywords:
                    if k in string:
                        cell_info = [k, coordinates, string_list]
                        useful_cells.append(cell_info)
                        
                # --- Masking tables --- #
                cv2.rectangle(table_bgr2, (x, y), (x+w, y+h), (0, 0, 0), -1)
        
        table_mask = cv2.cvtColor(table_bgr2, cv2.COLOR_BGR2GRAY)
        table_mask = cv2.dilate(np.bitwise_not(table_mask), rect_kernel3, iterations=5)
        
        drawing = np.bitwise_not(bin_img) + table_mask
        drawing[drawing >= 5] = 255
        drawing[drawing < 5] = 0
        
        # --- Checking for unattended full-vertical tables --- #
        _, bin_drawing = cv2.threshold(drawing, 150, 255, cv2.THRESH_BINARY_INV)
        bin_drawing = cv2.erode(bin_drawing, horiz_kernel, iterations = 5)
        bin_drawing = cv2.dilate(bin_drawing, horiz_kernel, iterations = 5)
        
        vertical_lines_dil = cv2.dilate(bin_drawing, rect_kernel3, iterations = 2)
        vert_contours, hierarchy = cv2.findContours(vertical_lines_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        vert_tf = False
        for c in vert_contours:
            x, y, w, h = cv2.boundingRect(c)
            if (h >= nrow - 50):
                vert_tf = True
                break
        
        # --- Extracting with largest contour --- #
        if (vert_tf == True and len(useful_cells) == 0):
            drawing_mask2 = np.zeros((nrow, ncol), dtype=np.uint8)
            
            contours, _ = cv2.findContours(np.bitwise_not(bin_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse = True)
            x, y, w, h = cv2.boundingRect(sorted_contours[0])
            drawing_mask2[y:y+h, x:x+w] = 255
            
            # Extracting drawing number
            drawingNum = GetString(init_img, "DRAWING NUMBER", "DRAWING NO")
            
            if (len(drawingNum) > 0):
                useful_cells.append(["DRAWING NUMBER", None, ["", drawingNum]])
        
        # --- Sorting and processing data --- #
        # Remove duplicate cell data
        table_data = []
        for c in useful_cells:
            if c not in table_data:
                table_data.append(c)

        # Sorting titles in alphabetical order
        table_data.sort(key=lambda x: x[0])
        
        # Process amendments separately
        amendments = None
        amend_index = [i for i, item in enumerate(table_data) if item[0] == "AMENDMENTS"]
        
        if amend_index:
            amendments = table_data.pop(amend_index[0])
        
        # Convert table data to dictionary
        image_json_data = {}
        for info in table_data:
            image_json_data[info[0]] = info[2][1] if len(info[2]) > 1 else ""
        
        # Add amendments if exists
        if amendments:
            image_json_data['AMENDMENTS'] = amendments[2][1:]
        
        # Save extracted image
        extracted_image_path = os.path.join('extracted', f'drawing_{image_number}.png')
        cv2.imwrite(extracted_image_path, drawing)
        extracted_paths.append(extracted_image_path)
        
        # Save JSON data
        json_data[image_number] = image_json_data
        
        # Save JSON to file
        json_file_path = os.path.join('json_format', f'data_{image_number}.json')
        with open(json_file_path, 'w') as f:
            json.dump(image_json_data, f, indent=4)
    
    return json_data, extracted_paths


# def main():
#     # Add sidebar with developer information
#     st.sidebar.title("About the Project")
#     st.sidebar.markdown("""
#     **Developed by:**
#     - Jhanani
#     - Balaji

#     **Purpose:** 
#     Hackathon Project - Automotive Drawing Analysis
#     """)
    
#     st.title("Automotive Drawing Analysis")
    
#     # SVG Link
#     svg_link = "https://www.mermaidchart.com/raw/bdc8c1d9-9f78-4b67-b9b3-5f36e6f489e7?theme=light&version=v0.1&format=svg"          
    
#     # Display the SVG link as a clickable hyperlink
#     st.markdown(f"[View SVG Diagram]({svg_link})", unsafe_allow_html=True)
    
#     # Rest of your existing code...
#     # File uploader
#     uploaded_files = st.file_uploader("Upload Technical Drawings", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
#     if uploaded_files:
#         # Process images
#         json_data, extracted_paths = process_images(uploaded_files)
        
#         # Display extracted data
#         st.subheader("Extracted Information")
#         st.json(json_data)
        
#         # Display extracted images and AI analysis
#         st.subheader("Drawing Analysis")
        
#         for i, (file, extracted_path, data) in enumerate(zip(uploaded_files, extracted_paths, json_data.values())):
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.image(extracted_path, caption=f'Extracted {file.name}')
            
#             with col2:
#                 # Analyze drawing with GPT
#                 ai_description = analyze_drawing_with_gpt(extracted_path, data)
#                 st.markdown("**AI Technical Analysis:**")
#                 st.write(ai_description)

# if __name__ == "__main__":
#     main()


def main():
    # SVG Link
    svg_link = "https://www.mermaidchart.com/raw/bdc8c1d9-9f78-4b67-b9b3-5f36e6f489e7?theme=light&version=v0.1&format=svg"
    
    # Add sidebar with developer information and SVG link
    st.sidebar.title("About the Project")
    st.sidebar.markdown("""
    **Developed by:**
    - Jhanani
    - Balaji

    **Purpose:** 
    Hackathon Project - Automotive Drawing Analysis
    """)
    
    # Add SVG link to sidebar
    st.sidebar.markdown(f"**Project Diagram:**\n[View SVG Diagram]({svg_link})", unsafe_allow_html=True)
    
    st.title("Automotive Drawing Analysis")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload Technical Drawings", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        # Process images
        json_data, extracted_paths = process_images(uploaded_files)
        
        # Display extracted data
        st.subheader("Extracted Information")
        st.json(json_data)
        
        # Display extracted images and AI analysis
        st.subheader("Drawing Analysis")
        
        for i, (file, extracted_path, data) in enumerate(zip(uploaded_files, extracted_paths, json_data.values())):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(extracted_path, caption=f'Extracted {file.name}')
            
            with col2:
                # Analyze drawing with GPT
                ai_description = analyze_drawing_with_gpt(extracted_path, data)
                st.markdown("**AI Technical Analysis:**")
                st.write(ai_description)

if __name__ == "__main__":
    main()
