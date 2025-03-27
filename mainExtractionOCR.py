import base64                            
import openai
import streamlit as st
import cv2
import os
import pytesseract
import numpy as np
import json
import importlib.util
import pandas as pd
from neo4j import GraphDatabase
import os
from roboflow import Roboflow
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG)

OPENAI_API_KEY = "#Enter actual API key" 
openai.api_key = OPENAI_API_KEY

# Hardcoded Neo4j Credentials
NEO4J_URI = "neo4j+ssc://d0f3d58e.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gMS9dY28yuUc3x83w5E4Fkrzp5AHKvAOhzLjd1vWt_4"

ROBOFLOW_API_KEY = "J3HiZxfxzJQz1Vh7ND2h"
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("symbol_seg")
model = project.version(2).model

def init_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Test the connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        return None

# Initialize Neo4j driver
driver = init_neo4j()

def create_nodes_and_relationships(tx, properties):
    """Creates multiple nodes and establishes relationships using CREATE."""
    query = """
    CREATE (p:Project {project_no: $project_no, status: $status, title: $title, unit: $unit})
    CREATE (d:Drawing {drawing_no: $drawing_no, cad_no: $cad_no, page: $page, lang: $lang, image_name: $image_name})
    CREATE (c:Contractor {name: $contractor})
    CREATE (ch:Person {name: $checked_by})
    CREATE (dr:Person {name: $drawn_by})
    CREATE (a:Approval {status: $approved})
    CREATE (am:Amendment {details: $amendments})

    CREATE (p)-[:HAS_DRAWING]->(d)
    CREATE (c)-[:ASSIGNED_TO]->(p)
    CREATE (ch)-[:CHECKED]->(d)
    CREATE (dr)-[:DRAWN]->(d)
    CREATE (d)-[:HAS_APPROVAL]->(a)
    CREATE (d)-[:HAS_AMENDMENT]->(am)
    """
    
    tx.run(query, 
        project_no=properties.get("Project_NO", "Unknown"),
        status=properties.get("STATUS", "Unknown"),
        title=properties.get("TITLE", "Unknown"),
        unit=properties.get("UNIT", "Unknown"),
        drawing_no=properties.get("DRAWING_NUMBER", "Unknown"),
        cad_no=properties.get("CAD_NO", "Unknown"),
        page=properties.get("PAGE", "Unknown"),
        lang=properties.get("LANG", "Unknown"),
        image_name=properties.get("Image_Name", "Unknown"),
        contractor=properties.get("CONTRACTOR", "Unknown"),
        checked_by=properties.get("CHECKED_BY", "Unknown"),
        drawn_by=properties.get("DRAWN_BY", "Unknown"),
        approved=properties.get("APPROVED", "Unknown"),
        amendments=properties.get("AMENDMENTS", "None")
    )

def insert_data_to_neo4j(df, driver):
    """Processes DataFrame and inserts data into Neo4j."""
    try:
        with driver.session() as session:
            for _, row in df.iterrows():
                properties = row.dropna().to_dict()
                session.execute_write(create_nodes_and_relationships, properties)
        return True
    except Exception as e:
        st.error(f"Error inserting data to Neo4j: {str(e)}")
        return False
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
        response = openai.chat.completions.create(
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

def process_images(uploaded_files):
    # Ensure output directories exist
    os.makedirs('extracted', exist_ok=True)

    # Store data for Excel and Streamlit
    excel_data = []
    extracted_paths = []

    for image_file in uploaded_files:
        # Reads the uploaded image as a NumPy array. Converts it to grayscale for processing.
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
            
            # Add null check before length check
            if drawingNum is not None and len(drawingNum) > 0:
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
        
        # Convert table data to dictionary and prepare for Excel
        image_data = {"Image Name": filename}
        for info in table_data:
            image_data[info[0]] = info[2][1] if len(info[2]) > 1 else ""
        
        # Add amendments if exists
        if amendments:
            image_data['AMENDMENTS'] = amendments[2][1:]
        
        # Save extracted image
        extracted_image_path = os.path.join('extracted', f'drawing_{image_number}.png')
        cv2.imwrite(extracted_image_path, drawing)
        extracted_paths.append(extracted_image_path)
        
        # Add Roboflow symbol detection
        try:
            # Initial symbol detection
            symbol_predictions = trigger_roboflow_detection(extracted_image_path)
            image_data["symbols_detected"] = symbol_predictions
        except Exception as e:
            st.warning(f"Initial symbol detection failed: {str(e)}")
            image_data["symbols_detected"] = None

        # Add to excel data list
        excel_data.append(image_data)
    
    # Save to Excel 
    df = pd.DataFrame(excel_data)
    excel_path = 'extracted_info.xlsx'
    df.to_excel(excel_path, index=False)
    
    return excel_data, extracted_paths

def trigger_roboflow_detection(drawing_path):
    """
    Trigger Roboflow symbol detection on a specific drawing path
    """
    try:
        predictions = model.predict(drawing_path, confidence=40)
        # Convert predictions to list of dictionaries if needed
        if isinstance(predictions, str):
            return []
        pred_list = predictions.json() if hasattr(predictions, 'json') else predictions
        logging.debug(f"Roboflow predictions: {pred_list}")
        return pred_list
    except Exception as e:
        logging.error(f"Symbol detection error: {str(e)}")
        return []

def trigger_detailed_roboflow_detection(drawing_path, confidence_threshold=40):
    """
    Trigger a more detailed Roboflow symbol detection with comprehensive analysis
    """
    try:
        predictions = model.predict(drawing_path, confidence=confidence_threshold)
        pred_list = predictions.json() if hasattr(predictions, 'json') else predictions
        
        if not pred_list:
            return None
            
        # Analyze predictions
        analysis = {
            'total_predictions': len(pred_list),
            'predictions': pred_list,
            'confidence_stats': {
                'min': min((p.get('confidence', 0) for p in pred_list), default=0),
                'max': max((p.get('confidence', 0) for p in pred_list), default=0),
                'average': sum(p.get('confidence', 0) for p in pred_list) / len(pred_list) if pred_list else 0
            },
            'unique_classes': list(set(p.get('class', '') for p in pred_list))
        }
        
        logging.debug(f"Detailed Roboflow predictions: {analysis}")
        return analysis
    except Exception as e:
        logging.error(f"Detailed symbol detection error: {str(e)}")
        return None

def display_predictions(predictions):
    """Helper function to safely display predictions"""
    if not predictions:
        return
        
    for prediction in predictions:
        try:
            class_name = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            st.write(f"- {class_name}: {confidence:.2f}%")
        except Exception as e:
            logging.error(f"Error displaying prediction: {str(e)}")
            continue

def main():
    # SVG Link
    svg_link = "https://www.mermaidchart.com/raw/bdc8c1d9-9f78-4b67-b9b3-5f36e6f489e7?theme=light&version=v0.1&format=svg"
    
    # Add sidebar with developer information and SVG link
    st.sidebar.title("About the Project")
    st.sidebar.markdown("""
    ***Developed by:***
    - Jhanani
    - Balaji
    - Shakthi
    - Swetha                                        

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
        excel_data, extracted_paths = process_images(uploaded_files)
        
        # Create DataFrame and clean column names
        df = pd.DataFrame(excel_data)
        df.columns = df.columns.str.replace(r'\W+', '_', regex=True)
        
        # Display extracted data
        st.subheader("Extracted Information")
        st.dataframe(df)
        
        # Save to Excel and provide download
        excel_path = 'extracted_info.xlsx'
        df.to_excel(excel_path, index=False)
        
        with open(excel_path, 'rb') as f:
            excel_bytes = f.read()
        st.download_button(
            label="Download Excel File", 
            data=excel_bytes, 
            file_name='extracted_info.xlsx', 
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        # Insert data into Neo4j
        if st.button("Store in Neo4j Database"):
            with st.spinner("Inserting data into Neo4j..."):
                if insert_data_to_neo4j(df, driver):
                    st.success("Data successfully stored in Neo4j!")
        
        # Display extracted images and AI analysis
        st.subheader("Drawing Analysis")
        
        for i, (file, extracted_path, data) in enumerate(zip(uploaded_files, extracted_paths, excel_data)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(extracted_path, caption=f'Extracted {file.name}')
            
            with col2:
                # Analyze drawing with GPT
                ai_description = analyze_drawing_with_gpt(extracted_path, data)
                st.markdown("**AI Technical Analysis:**")
                st.write(ai_description)
                
                # Add symbol detection button
                if st.button(f"Detect Symbols in {file.name}", key=f"roboflow_btn_{i}"):
                    with st.spinner("Detecting Symbols..."):
                        symbol_predictions = trigger_roboflow_detection(extracted_path)
                        
                        if symbol_predictions:
                            st.markdown("**Detected Symbols:**")
                            display_predictions(symbol_predictions)
                        else:
                            st.warning("No symbols detected or detection failed.")
                
                # Show previously detected symbols if available
                if data.get("symbols_detected"):
                    st.markdown("**Previously Detected Symbols:**")
                    predictions = data["symbols_detected"]
                    display_predictions(predictions)
            
            # Add an expandable popup for detailed symbol detection
            with st.expander(f"Advanced Symbol Detection for {file.name}"):
                # Confidence threshold slider
                confidence_threshold = st.slider(
                    "Confidence Threshold", 
                    min_value=0, 
                    max_value=100, 
                    value=40, 
                    key=f"confidence_slider_{i}"
                )
                
                # Detailed detection button
                if st.button(f"Run Detailed Detection", key=f"detailed_detection_{i}"):
                    with st.spinner("Running Advanced Symbol Detection..."):
                        detailed_predictions = trigger_detailed_roboflow_detection(
                            extracted_path, 
                            confidence_threshold
                        )
                        
                        if detailed_predictions:
                            # Display comprehensive analysis
                            st.markdown("**Detailed Symbol Detection Report**")
                            
                            # Statistics
                            st.write(f"**Total Predictions:** {detailed_predictions['total_predictions']}")
                            
                            # Confidence stats
                            st.markdown("**Confidence Statistics:**")
                            stats = detailed_predictions['confidence_stats']
                            st.write(f"- Min: {stats['min']:.2f}%")
                            st.write(f"- Max: {stats['max']:.2f}%")
                            st.write(f"- Avg: {stats['average']:.2f}%")
                            
                            # Classes
                            st.markdown("**Unique Symbol Classes:**")
                            st.write(", ".join(detailed_predictions['unique_classes']))
                            
                            # Individual predictions
                            st.markdown("**Individual Symbols:**")
                            for pred in detailed_predictions['predictions']:
                                st.write(
                                    f"- **{pred['class']}**: "
                                    f"Confidence: {pred['confidence']:.2f}%, "
                                    f"Location: ({pred['x']:.0f}, {pred['y']:.0f})"
                                )
                        else:
                            st.warning("Advanced detection failed or no symbols found.")

if __name__ == "__main__":
    main()