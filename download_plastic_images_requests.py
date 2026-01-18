import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
if not PEXELS_API_KEY:
    raise ValueError("❌ PEXELS_API_KEY not found in .env file.")

# Config
QUERY = "floating plastic waste in sea -people"

"river plastic garbage no humans"

"plastic trash in water environment -person -child"

"polluted water plastic floating waste"
"plastic debris in ocean without people"

IMAGES_TO_DOWNLOAD = 200
OUTPUT_DIR = "downloaded_plastic_images"
PER_PAGE = 15  # Max allowed by Pexels
HEADERS = {"Authorization": PEXELS_API_KEY}
URL = "https://api.pexels.com/v1/search"

os.makedirs(OUTPUT_DIR, exist_ok=True)

downloaded = 0
page = 1

while downloaded < IMAGES_TO_DOWNLOAD:
    params = {
        "query": QUERY,
        "per_page": PER_PAGE,
        "page": page
    }

    response = requests.get(URL, headers=HEADERS, params=params)
    data = response.json()

    photos = data.get("photos", [])
    if not photos:
        print("No more photos found. - download_plastic_images_requests.py:45")
        break

    for photo in photos:
        if downloaded >= IMAGES_TO_DOWNLOAD:
            break
        img_url = photo["src"]["large"]
        img_data = requests.get(img_url).content
        filename = os.path.join(OUTPUT_DIR, f"image_{downloaded+1:03d}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        downloaded += 1
        print(f"✅ Downloaded {filename} - download_plastic_images_requests.py:57")

    page += 1

print(f"\n✅ Done. {downloaded} images saved in '{OUTPUT_DIR}' - download_plastic_images_requests.py:61")
