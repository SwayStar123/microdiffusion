import os
import pyarrow.parquet as pq
from datasets import Dataset
from datasets import load_dataset
import torch

dataset = load_dataset("../../datasets/commoncatalog_cc_by_moondream_latents", split="train")
print("Dataset length: ", len(dataset))

print(len(dataset[0]["embedding"]))

id = "09821753"

print("Image id: ", dataset[542]["image_id"])
print("Latent shape: ", dataset[542]["latent_shape"])
print("Actual latent shape: ", torch.tensor(dataset[542]["latent"]).shape)