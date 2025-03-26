#------------------- this is drawingNum.py---------------------------

import cv2
import pytesseract
from matplotlib import pyplot as pt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# -----------------------------------------------------------------------------------------------------------------------

# Page segmentation mode, PSM was changed to 6 since each page is a single uniform text block.
def GetString(technical_drawing, keyword1, keyword2):
    copy = technical_drawing.copy()
    [nrow, ncol]= technical_drawing.shape  #Stores the number of rows (nrow) and columns (ncol) of the image.


# Why?
# OCR works better on clean images without small pixel variations.
# If skipped, noise might interfere with contour detection and text extraction.    

    blur = cv2.GaussianBlur(copy, (3,3), 0)  #Applies Gaussian blur to reduce noise and make the image smoother.
                                             #here, 0 is standard deviation and When set to 0, OpenCV automatically calculates it based on the kernel size.



# -----------------------------------------------------------------------------------------------------------------------
# Why?
# This helps in detecting boundaries (edges) clearly.
# If skipped, text might blend into the background, making extraction harder.
# 127 → Threshold value: Pixels above 127 become 1(black), and those below become 0 (white).

    ret, thresh = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY_INV)  #Converts the image into black (text) and white (background) for easier text detection.

# -------------------------------------------------------------------------------------------------------------------------    
# Why?
#     Contours help locate regions of interest (e.g., text inside a box).
# If skipped, there would be no way to isolate the drawing number region.

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #Finds contours in the binary image using cv2.findContours().
                                                                                            # cv2.RETR_TREE → Retrieves all contours and maintains a hierarchy (useful for nested objects like boxes)..
                                                                                            # cv2.CHAIN_APPROX_SIMPLE: Compresses contour points for efficiency.

# -------------------------------------------------------------------------------------------------------------------------
# Why?
# Helps filter out small, irrelevant contours.
# If skipped, we might process unwanted noise instead of the text box.    

    for contour in contours:                           #Loops through the contours found in the image.
        area = cv2.contourArea(contour)                # Calculates the area of each contour.

# -------------------------------------------------------------------------------------------------------------------------
# Why?
# Ensures we only process reasonably-sized text boxes.
# If this condition is too strict, the text might be missed.
# If too loose, we might include irrelevant regions.    
    
        if (area > 40000 and area < 5000000):           #If the contour's area is between 40,000 and 5,000,000 pixels, it is considered relevant.

# Why?
# This step isolates only the relevant text.
# If skipped, OCR might process the entire image, leading to errors.


            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(technical_drawing, (x,y), (x+w, y+h), (36,255,12), -1)
            ROI = copy[y:y+h, x:x+w]  

# copy[y:y+h, x:x+w] → Crops a Region of Interest (ROI) from copy.
# Why y:y+h and x:x+w?
# y:y+h → Selects the vertical range of pixels from y to y+h.
# x:x+w → Selects the horizontal range of pixels from x to x+w.

# This extracts the detected contour as a separate image.            

# -------------------------------------------------------------------------------------------------------------------------

# Uses Tesseract to extract text from the ROI.
# --psm 6 → Tells Tesseract that the region contains a single block of text.            
            string = (pytesseract.image_to_string(ROI, config ='--psm 6')).strip()
            if (string == ""):
                return
            import os

            # Create a 'visualizations' folder if it doesn't exist
            if not os.path.exists('visualizations'):
                os.makedirs('visualizations')

            # Generate a unique filename
            filename = f"visualizations/ROI_{x}_{y}_{w}_{h}.png"

            # Create the figure and save it
            pt.figure()
            pt.imshow(ROI, cmap="gray")
            pt.axis('off')  # Turn off axis labels
            pt.savefig(filename, bbox_inches='tight', pad_inches=0)
            pt.close()  # Close the figure to free up memory

# -------------------------------------------------------------------------------------------------------------------------

# Iterates through string list, checking if any element contains keyword1 or keyword2.
# If a match is found, sets indexOfValue = i (stores the index where the keyword was found).
# While extracted_string is empty, moves to the next index (indexOfValue + 1).
# If indexOfValue is still within range (indexOfValue < len(string)), assigns the next string value to extracted_string.
# If out of range, it exits (break).

            if (keyword1 in string or keyword2 in string): #Checks if the extracted text contains either keyword1 or keyword2 (e.g., "DRAWING NUMBER").
                ROI = copy[y:y+h+100, x+10:x+w]  # we take a larger area of the box identified - Extracts the region containing the text.  
    
                copyROI = ROI.copy()
                ret, thresh = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY_INV)
                
                #--- Remove any potential boxes surrounding the letters which can impair extraction through OCR ---#

# Why?
# Lines in tables or boxes reduce OCR accuracy.
# If skipped, OCR might detect partial letters or garbage text.
                # Remove horizontal lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
                remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(copyROI, [c], -1, (255,255,255), 5)
                
                # Remove vertical lines
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
                remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(copyROI, [c], -1, (255,255,255), 5)
                
# ------------------------------------- Final Reading of Box ------------------------------ --
# Why?
# Extracts text after removing lines.
# splitlines() → Splits text into separate lines.

                string = (pytesseract.image_to_string(copyROI, config ='--psm 6')).strip()    
                string = string.splitlines()
                extracted_string = ""   #If the extracted text is empty, the function returns immediately.



                for i in range(len(string)):
                    
                    if keyword1 in string[i] or keyword2 in string[i]:  #Checks if the extracted text contains either keyword1 or keyword2 (e.g., "DRAWING NUMBER").
                        indexOfValue = i
                        while extracted_string == "":     
                            indexOfValue = indexOfValue + 1
                            if ((indexOfValue) < len(string)): # if true means that this string has this index
                                extracted_string = string[indexOfValue]
                            else:
                                break
                        return extracted_string
                
                return extracted_string  #Returns the final extracted value.


                break
   
# img = cv2.imread("08.png", 0)     
# data_extract = {}     
  
# drawingNum = GetString(img, "DRAWING NUMBER", "DRAWING NO")
# drawnBy = GetString(img, "DRAWN BY", "DRAWN")
# checkedBy = GetString(img, "CHECKED BY", "CHECKED")
# title = GetString(img, "TITLE", "DRAWING TITLE")
# approvedBy = GetString(img, "APPPROVED BY", "APPROVED")
# contractor = GetString(img, "CONTRACTOR", "COMPANY")
# unit = GetString(img, "UNIT", "UNIT")
# status = GetString(img, "STATUS", "STATUS")
# page = GetString(img, "PAGE", "PAGE")
# projectNum = GetString(img, "PROJECT NO", "PROJECT NUM")
# lang = GetString(img, "LANG", "LANG")
# cad = GetString(img, "CAD NO", "CAD")
# font = GetString(img, "FONT", "FONT STYLE")

# data_extract["drawing number"] = drawingNum
# data_extract["drawn by"] = drawnBy
# data_extract["checked by"] = checkedBy
# data_extract["title"] = title
# data_extract["approved by"] = approvedBy
# data_extract["contractor"] = contractor
# data_extract["unit"] = unit
# data_extract["status"] = status
# data_extract["page"] = page
# data_extract["projectNum"] = projectNum
# data_extract["lang"] = lang
# data_extract["cad"] = cad
# data_extract["font"] = font

# print(data_extract)