import cv2
import pytesseract
from matplotlib import pyplot as pt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Page segmentation mode, PSM was changed to 6 since each page is a single uniform text block.
    
def GetString(technical_drawing, keyword1, keyword2):
    copy = technical_drawing.copy()
    [nrow, ncol]= technical_drawing.shape  #Stores the number of rows (nrow) and columns (ncol) of the image.

# Why?
# OCR works better on clean images without small pixel variations.

# If skipped, noise might interfere with contour detection and text extraction.    

    blur = cv2.GaussianBlur(copy, (3,3), 0)  #Applies Gaussian blur to reduce noise and make the image smoother.

# Why?
# This helps in detecting boundaries (edges) clearly.
# If skipped, text might blend into the background, making extraction harder.

    ret, thresh = cv2.threshold(blur, 127, 1, cv2.THRESH_BINARY_INV)  #Converts the image into black (text) and white (background) for easier text detection.
    
# Why?
#     Contours help locate regions of interest (e.g., text inside a box).
# If skipped, there would be no way to isolate the drawing number region.

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #Finds contours in the binary image using cv2.findContours().
                                                                                            # cv2.RETR_TREE → Retrieves all contours and maintains a hierarchy (useful for nested objects like boxes)..
                                                                                            # cv2.CHAIN_APPROX_SIMPLE: Compresses contour points for efficiency.

# Why?
# Helps filter out small, irrelevant contours.
# If skipped, we might process unwanted noise instead of the text box.    

    for contour in contours:                           #Loops through the contours found in the image.
        area = cv2.contourArea(contour)                # Calculates the area of each contour.

# Why?
# Ensures we only process reasonably-sized text boxes.
# If this condition is too strict, the text might be missed.
# If too loose, we might include irrelevant regions.    
    
        if (area > 40000 and area < 5000000):           #If the contour's area is between 40,000 and 5,000,000 pixels, it is considered relevant.
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(technical_drawing, (x,y), (x+w, y+h), (36,255,12), -1)
            ROI = copy[y:y+h, x:x+w]
            
            string = (pytesseract.image_to_string(ROI, config ='--psm 6')).strip()
            if (string == ""):
                return
            
            # pt.figure()
            # pt.imshow(ROI, cmap = "gray")
            
            if (keyword1 in string or keyword2 in string):
                ROI = copy[y:y+h+100, x+10:x+w]  # we take a larger area of the box identified - Extracts the region containing the text.  
    
                copyROI = ROI.copy()
                ret, thresh = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY_INV)
                
                #--- Remove any potential boxes surrounding the letters which can impair extraction through OCR ---#
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
                
                # --- Final Reading of Box --- #
                # Uses Tesseract OCR to extract text from ROI.


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
                
                return extracted_string
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