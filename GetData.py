## From Data
import pandas as pd
import numpy as np
## Web Scraping 
import requests
from bs4 import BeautifulSoup
## Image manipulation
from PIL import Image
## Filing Functionality 
import io
import os
import hashlib

artLoc = pd.read_csv("./painting_dataset_2018.csv")
#print(artLoc.describe())

print(artLoc['Image URL'][0])
#result = requests.get(artLoc['Image URL'][0])


def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

persist_image(r"C:\Users\Admin\Desktop\ART", "https://d3d00swyhr67nd.cloudfront.net/w1200h1200/collection/SFK/SED/SFK_SED_ST_1992_9_587-001.jpg")