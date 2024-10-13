# Load dataset loads images in random image_id order, so we need to index the image_ids to get the correct order
# This script indexes the image_ids and saves them to a file
from datasets import load_dataset
import json

def index_image_id_map_main():
    dataset = load_dataset("../../datasets/commoncatalog_cc_by_moondream_latents", split="train", streaming=True, columns=["image_id"])
    image_id_map = {}
    for i, image in enumerate(dataset):
        image_id_map[image["image_id"]] = i

    with open("../../datasets/commoncatalog_cc_by_moondream_metadata/image_id_map.json", "w") as fh:
        json.dump(image_id_map, fh)
    
if __name__ == "__main__": 
    index_image_id_map_main()