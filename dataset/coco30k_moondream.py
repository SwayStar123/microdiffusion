from datasets import load_dataset
import datasets
if __name__ == "__main__":   
    # Setup resolutions.pkl
    import pickle

    print("Loading dataset...")
    # Set offline mode
    datasets.config.HF_HUB_OFFLINE = 1 # Comment this out if you havent downloaded the dataset yet
    # Load the dataset
    dataset = load_dataset("sroecker/recap-coco30k-moondream", cache_dir="../../datasets/coco30k_moondream")

    print("Extracting image IDs and their corresponding resolutions...")
    # Extract image IDs and their corresponding resolutions
    res_map = {}

    for i, image in enumerate(dataset["train"]):
        res_map[i] = image["image"].size

    print("Saving resolutions.pkl...")
    with open("resolutions.pkl", "wb") as f:
        pickle.dump(res_map, f)

