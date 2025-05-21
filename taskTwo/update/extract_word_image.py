
from svgpathtools import svg2paths2, paths2svg
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import cv2
import glob #get files with pattern matching
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common import param_utils


# Cut out single words from scan file and save them as individual files

# 1. import svg file and image
# 2. extract paths and turn into "filled" polygons, calculate bounding boxes
# 3. apply to imag
# 4. binarization 
# (optional/todo) 5. further preprocessing: normalization, tilting, ...
# 6. export as new file


# Change these directory paths to match the expected structure
output_base = "./cutouts_png" 
os.makedirs(output_base, exist_ok=True)

# get the correct files
args = param_utils.parse_args_KWS()
pages = glob.glob(os.path.join(args.kwsTest, "images/*.jpg"))
cutouts = glob.glob(os.path.join(args.kwsTest, "locations/*.svg"))
pages.sort()
cutouts.sort()
for page, cutout in zip(pages, cutouts):
    page_number = os.path.splitext(os.path.basename(page))[0]
    print(f"Processing {page_number}...")
    
    # Create directory for this document
    doc_output_dir = os.path.join(output_base, page_number)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    paths, attributes, svg_attributes = svg2paths2(cutout)
    scan = cv2.imread(page, cv2.IMREAD_GRAYSCALE)
    
    for path, attribute in zip(paths, attributes):
        code = attribute['id']
        # Create correct filename - keep the document ID as part of filename
        filename = f"{page_number}-{code}"
        
        # Process as before...
        bbox = paths2svg.big_bounding_box(path)
        bbox = tuple(map(int, bbox))
        scan_crop = scan[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        
        # Binarize
        thresh = threshold_otsu(scan_crop)
        scan_crop_logic = scan_crop < thresh
        
        # Create mask
        height = int(bbox[3] - bbox[2])
        width = int(bbox[1] - bbox[0])
        
        mask = Image.new('1', (width, height), "black")
        mask_draw = ImageDraw.Draw(mask)
        polygon = []
        for edge in path:
            polygon.append((int(edge.point(0).real)-bbox[0], int(edge.point(0).imag)-bbox[2]))
        
        mask_draw.polygon(polygon, fill="white", outline=None)
        
        img = np.logical_and(scan_crop_logic, mask)
        img = np.invert(img)
        img = img.astype(float)
        img = cv2.resize(src=img, dsize=(width, 120), interpolation=cv2.INTER_NEAREST)
        img = img > 0
        img = Image.fromarray(img)
        
        # Save to the correct path with correct filename
        img.save(os.path.join(doc_output_dir, f"{filename}.png"))
        
    print(f"Done processing scan {os.path.basename(page)}")
