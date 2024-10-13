import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch
import os
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch.nn.functional as F
import numpy as np
import pickle
import time
from huggingface_hub import HfApi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import json

USERNAME = "KagakuAI"
DATASET_NAME = "commoncatalog_cc_by_moondream_latents"
METADATA_DATASET_NAME = "commoncatalog_cc_by_moondream_metadata"
IMG_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BATCH_SIZE_PER_GPU = 8
IMAGES_PER_PARQUET = BATCH_SIZE_PER_GPU * 100
CACHE_DIR_BASE = "../.."

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, max_size=(512,512), divisible=16, min_dim=256, base_res=(512,512), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def get_ideal_resolution(self, image_size) -> tuple[int, int]:
        w, h = image_size
        aspect = float(w)/float(h)
        bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
        return self.resolutions[bucket_id]

def resize_and_crop(image, target_size):
    # image: PIL Image
    # target_size: (width, height)
    target_w, target_h = target_size
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_w / target_h
    if abs(aspect_ratio - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, resize directly
        image = image.resize((target_w, target_h), Image.BICUBIC)
    else:
        # Resize while preserving aspect ratio, then random crop
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target, resize height to target height
            new_height = target_h
            new_width = int(aspect_ratio * new_height)
        else:
            # Image is taller than target, resize width to target width
            new_width = target_w
            new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        # Random crop to target size
        left = np.random.randint(0, new_width - target_w + 1)
        upper = np.random.randint(0, new_height - target_h + 1)
        image = image.crop((left, upper, left + target_w, upper + target_h))
    return image

def preprocess_image(image):
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to tensor, normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1], shape (C,H,W)
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
    ])
    image_tensor = transform(image)
    return image_tensor

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank (int): the rank of the current process
        world_size (int): number of processes participating in the job
    """
    addr = "localhost"
    port = "12355"
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

    init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=f"tcp://{addr}:{port}?use_libuv=0")

def calculate_latents_and_embeddings(batch, vae, siglip_model, siglip_tokenizer, device, bucket_manager, moondream_model, moondream_tokenizer):
    """Function to calculate latents and embeddings for a given image."""
    images = batch[IMG_COLUMN_NAME]
    keys = batch[IMAGE_ID_COLUMN_NAME]
    target_resolutions = [bucket_manager.get_ideal_resolution(image.size) for image in images]
    resized_images = [resize_and_crop(image, target_res) for image, target_res in zip(images, target_resolutions)]

    # Can only batch process in VAE if resolutions are same, queue the same resolutions
    queue = {}
    for i, image in enumerate(resized_images):
        if image.size in queue:
            queue[image.size][keys[i]] = image
        else:
            queue[image.size] = {keys[i]: image}

    res_latents = {}
    res_captions = {}
    for resolution, batch in queue.items():
        ks = batch.keys()
        imgs = list(batch.values())

        captions = moondream_model.batch_answer(
            images=imgs,
            captions=["Caption this image."] * len(imgs),
            tokenizer=moondream_tokenizer,
        )
        captions_list = [caption for caption in captions]
        for key, caption in zip(ks, captions_list):
            res_captions[key] = caption

        imgs = [preprocess_image(image) for image in imgs]

        # Convert list of imgs to tensor of batch images
        imgs = torch.stack(imgs, dim=0).to(device)
        latents = vae.encode(imgs).latent_dist.sample().cpu()
        latents_list = [latent for latent in latents]
        for key, latent in zip(ks, latents_list):
            res_latents[key] = latent

    # Reorder latents based on keys
    latents = [res_latents[key] for key in keys]
    captions = [res_captions[key] for key in keys]

    texts = siglip_tokenizer(captions, context_length=siglip_model.context_length).to(device)
    text_embeddings = siglip_model.encode_text(texts)

    return latents, text_embeddings.cpu(), captions

def create_schema():
    """Create schemas for latents and embeddings parquet tables."""
    latents_schema = pa.schema([
        ('image_id', pa.string()),
        ('latent', pa.list_(pa.float32())),
        ('latent_shape', pa.list_(pa.int64())),
        ('embedding', pa.list_(pa.float32())),
        ('caption', pa.string())
    ])
    return latents_schema

def write_parquet(latents_list, latents_parquet_file, latents_schema):
    """Write latents and embeddings data into parquet files."""
    latents_table = pa.Table.from_pydict({
        'image_id': [item['image_id'] for item in latents_list],
        'latent': [item['latent'] for item in latents_list],
        'latent_shape': [item['latent_shape'] for item in latents_list],
        'embedding': [item['embedding'] for item in latents_list],
        'caption': [item['caption'] for item in latents_list]
    }, schema=latents_schema)

    pq.write_table(latents_table, latents_parquet_file)

# def upload_and_delete_files(api, latents_parquet_file, rank, index):
#     print(f"Uploading parquet for rank {rank} and index {index}")
#     # Upload files
#     api.upload_file(
#         path_or_fileobj=latents_parquet_file,
#         path_in_repo=f"{rank}/latents/{index}_latents.parquet",
#         repo_id=f"{USERNAME}/{DATASET_NAME}",
#         repo_type="dataset",
#     )

#     # Delete local files
#     os.remove(latents_parquet_file)

def process_images(rank: int, world_size: int, dataset, vae, siglip_model, tokenizer, bucket_manager, api, moondream_model, moondream_tokenizer):
    ddp_setup(rank, world_size)
    dataset = split_dataset_by_node(dataset, rank, world_size).batch(BATCH_SIZE_PER_GPU)

    device = torch.device(f'cuda:{rank}')
    vae = vae.to(device)
    siglip_model = siglip_model.to(device)
    moondream_model = moondream_model.to(device)

    image_id_res_map = {}
    image_id_caption_map = {}
    latents_list = []

    executor = ThreadPoolExecutor(max_workers=8)  # Adjust the number of threads as needed
    latents_schema = create_schema()

    # Create directories if they don't exist
    latents_dir = f"{CACHE_DIR_BASE}/datasets/{DATASET_NAME}/{rank}/latents"
    os.makedirs(latents_dir, exist_ok=True)
    index = 0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        progress_bar = tqdm(desc=f"Approximate processed", unit="img") if rank == 0 else None
        for i, batch in enumerate(dataset):
            # Calculate latents and embeddings
            latents, text_embeddings, captions = calculate_latents_and_embeddings(batch, vae, siglip_model, tokenizer, device, bucket_manager, moondream_model, moondream_tokenizer)
            image_ids = batch[IMAGE_ID_COLUMN_NAME]
            assert len(latents) == BATCH_SIZE_PER_GPU, f"Latents length mismatch: {len(latents)} != {BATCH_SIZE_PER_GPU}"
            assert len(text_embeddings) == BATCH_SIZE_PER_GPU, f"Text embeddings length mismatch: {len(text_embeddings)} != {BATCH_SIZE_PER_GPU}"
            assert len(captions) == BATCH_SIZE_PER_GPU, f"Captions length mismatch: {len(captions)} != {BATCH_SIZE_PER_GPU}"

            for image_id, latent, text_embedding, caption in zip(image_ids, latents, text_embeddings, captions):
                image_id_res_map[image_id] = latent.shape[1:]
                image_id_caption_map[image_id] = caption

                # Append to the lists
                latents_list.append({
                    'image_id': str(image_id),
                    'latent': latent.numpy().flatten().tolist(),
                    'latent_shape': list(latent.shape),
                    'embedding': text_embedding.numpy().flatten().tolist(),
                    'caption': caption
                })

            # Write Parquet files after reaching the IMAGES_PER_PARQUET threshold
            if (i + 1) % (IMAGES_PER_PARQUET//BATCH_SIZE_PER_GPU) == 0:
                latents_parquet_file = f"{latents_dir}/{index}_latents.parquet"

                write_parquet(latents_list, latents_parquet_file, latents_schema)

                # Submit the upload and delete task to the executor
                # Hugging face didnt increase repo limit so cant upload. Save dataset locally instead.
                # executor.submit(upload_and_delete_files, api, latents_parquet_file, rank, index)

                # Clear the lists for the next batch
                latents_list.clear()
                index += 1

            progress_bar.update(BATCH_SIZE_PER_GPU * world_size)

        # Handle remaining data after loop ends
        if latents_list:
            latents_parquet_file = f"{latents_dir}/{index}_latents.parquet"

            write_parquet(latents_list, latents_parquet_file, latents_schema)

            # Submit the upload and delete task to the executor
            # Hugging face didnt increase repo limit so cant upload. Save dataset locally instead.
            # executor.submit(upload_and_delete_files, api, latents_parquet_file, rank, index)

    # Wait for all uploads to complete
    executor.shutdown(wait=True)

    resmap_dir = f"{CACHE_DIR_BASE}/datasets/{METADATA_DATASET_NAME}/res_maps"
    caption_dir = f"{CACHE_DIR_BASE}/datasets/{METADATA_DATASET_NAME}/captions"
    os.makedirs(resmap_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)

    # Save res map
    with open(f"{resmap_dir}/{rank}_res_map.json", "w+") as fh:
        json.dump(image_id_res_map, fh)

    # Save caption map
    with open(f"{caption_dir}/{rank}_captions.json", "w+") as fh:
        json.dump(image_id_caption_map, fh)

    destroy_process_group()

def merge_res_and_caption_maps(filepath, world_size, api):
    """
    Merges the all the ranks' res_maps to a single file, and the all the ranks' captions to another single file.
    """
    res_map = {}
    captions = {}
    for i in range(world_size):
        with open(f"{filepath}/res_maps/{i}_res_map.json", "r") as fh:
            res_map.update(json.load(fh))
        with open(f"{filepath}/captions/{i}_captions.json", "r") as fh:
            captions.update(json.load(fh))

    with open(f"{filepath}/res_map.json", "w") as fh:
        json.dump(res_map, fh)
    with open(f"{filepath}/captions.json", "w") as fh:
        json.dump(captions, fh)

    # Upload res and caption maps
    api.upload_file(
        path_or_fileobj=f"{filepath}/res_map.json",
        path_in_repo=f"res_map.json",
        repo_id=f"{USERNAME}/{METADATA_DATASET_NAME}",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=f"{filepath}/captions.json",
        path_in_repo=f"captions.json",
        repo_id=f"{USERNAME}/{METADATA_DATASET_NAME}",
        repo_type="dataset",
    )

def preprocess_datasets_main(test=False):
    world_size = torch.cuda.device_count()

    if test:
        dataset = load_dataset("common-canvas/commoncatalog-cc-by", split="train", streaming=True, columns=["key", "jpg"]).take(1000)
    else:
        dataset = load_dataset("common-canvas/commoncatalog-cc-by", split="train", streaming=True, columns=["key", "jpg"])

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir=f"{CACHE_DIR_BASE}/models/vae")
    siglip_model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=f"{CACHE_DIR_BASE}/models/siglip")
    siglip_tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=f"{CACHE_DIR_BASE}/models/siglip")

    bucket_manager = BucketManager()
    api = HfApi()
    api.create_repo(repo_id=f"{USERNAME}/{METADATA_DATASET_NAME}", repo_type="dataset", exist_ok=True)

    model_id = "vikhyatk/moondream2"
    revision = "2024-07-23"
    moondream_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, cache_dir=f"{CACHE_DIR_BASE}/models/moondream"
    )
    moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=f"{CACHE_DIR_BASE}/models/moondream")

    mp.spawn(process_images, args=(world_size, dataset, vae, siglip_model, siglip_tokenizer, bucket_manager, api, moondream_model, moondream_tokenizer), nprocs=world_size)

    merge_res_and_caption_maps(f"{CACHE_DIR_BASE}/datasets/{METADATA_DATASET_NAME}", world_size, api)

if __name__ == "__main__":
    preprocess_datasets_main(test=True)