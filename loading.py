from pexels_api import API
import os
import urllib.request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str)
args = parser.parse_args()
query = args.query

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Chrome')]
urllib.request.install_opener(opener)

images_path = os.path.join("images", "0")
if not os.path.exists(images_path):
    os.makedirs(images_path)

API_KEY = os.environ.get("PEXELS_API_KEY")
api = API(API_KEY)
api.search(query)
processed = 0
number_of_images = 1000
while processed < number_of_images:
    photos = api.get_entries()
    for photo in photos:
        urllib.request.urlretrieve(photo.medium, os.path.join(images_path, f"{processed}.{photo.extension}"))
        processed += 1
        if processed >= number_of_images:
            break
    if not api.has_next_page:
        break
    api.search_next_page()
