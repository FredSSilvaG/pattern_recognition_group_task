import cv2
from xml.etree import ElementTree as ET
import numpy as np
import re
import os


# read image
def load_image(image_path):
    return cv2.imread(image_path, 0) 

# extract the words' polygon points 
def parse_word_polygons(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    polygons = {}
    for polygon in root.findall('.//svg:path', ns):
        word_id = polygon.attrib['id']
        d = polygon.attrib.get('d')
        if word_id and d:
            commands = re.findall(r'[ML]\s*([^MLZ]+)', d, flags=re.IGNORECASE)
            points = []
            for cmd in commands:
                coords = re.findall(r'[-+]?\d*\.?\d+', cmd)
                pairs = list(zip(coords[::2], coords[1::2]))
                for x, y in pairs:
                     points.append((int(round(float(x))), int(round(float(y))))) 
        polygons[word_id] = points
    return polygons

# crop word image
def crop_word_image(image, polygon):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)
    pts = np.array(polygon, dtype=np.int32)
    if pts.ndim == 2:
        pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    word_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    cropped_word = word_image[y:y+h, x:x+w]
    return cropped_word


# Crop all words from one page
def crop_all_words(image_path, svg_path, output_dir, doc_id):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    polygons = parse_word_polygons(svg_path)

    for wid, pts in polygons.items():
        crop = crop_word_image(image, pts)
        if crop.size > 0:
            fname = f"{wid}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), crop)
