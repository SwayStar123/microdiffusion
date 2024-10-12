import os
import pyarrow.parquet as pq
from datasets import Dataset
from datasets import load_dataset

dataset = load_dataset("../../datasets/commoncatalog_cc_by_moondream_latents", split="train")
print("Dataset length: ", len(dataset))