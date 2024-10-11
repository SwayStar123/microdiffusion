import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch
import os
from datasets import load_dataset
import datasets
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
from torch.utils.data import DataLoader
import json

USERNAME = "KagakuAI"
DATASET_NAME = "commoncatalog_cc_by_moondream_latents"
IMG_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BATCH_SIZE_PER_GPU = 8
IMAGES_PER_PARQUET = BATCH_SIZE_PER_GPU * 10000
CACHE_DIR_BASE = "../.."

"""
One common issue of existing image generation models is that they are very prone to producing images with unnatural crops. This is due to the fact that these models are trained to produce square images. However, most photos and artworks are not square. However, the model can only work on images of the same size at the same time, and during training, it is common practice to operate on multiple training samples at once to optimize the efficiency of the GPUs used. As a compromise, square images are chosen, and during training, only the center of each image is cropped out and then shown to the image generation model as a training example.

For example, humans are often generated without feet or heads, and swords consist of only a blade with a hilt and point outside the frame. As we are creating an image generation model to accompany our storytelling experience, it is important that our model is able to produce proper, uncropped characters, and generated knights should not be holding a metallic-looking straight line extending to infinity.

Another issue with training on cropped images is that it can lead to a mismatch between the text and the image.

For example, an image with a crown tag will often no longer contain a crown after a center crop is applied and the monarch has been, thereby, decapitated.

Knight wearing a crown with darkened regions removed by the center crop

We found that using random crops instead of center crops only slightly improves these issues.

Using Stable Diffusion with variable image sizes is possible, although it can be noticed that going too far beyond the native resolution of 512x512 tends to introduce repeated image elements, and very low resolutions produce indiscernible images.

Still, this indicated to us that training the model on variable sized images should be possible. Training on single, variable sized samples would be trivial, but also extremely slow and more liable to training instability due to the lack of regularization provided by the use of mini batches.

Custom Batch Generation
As no existing solution for this problem seems to exist, we have implemented custom batch generation code for our dataset that allows the creation of batches where every item in the batch has the same size, but the image size of batches may differ.

We do this through a method we call aspect ratio bucketing. An alternative approach would be to use a fixed image size, scale each image to fit within this fixed size and apply padding that is masked out during training. Since this leads to unnecessary computation during training, we have not chosen to follow this alternative approach.

In the following, we describe the original idea behind our custom batch generation scheme for aspect ratio bucketing.

First, we have to define which buckets we want to sort the images of our dataset into. For this purpose, we define a maximum image size of 512x768 with a maximum dimension size of 1024. Since the maximum image size is 512x768, which is larger than 512x512 and requires more VRAM, per-GPU batch size has to be lowered, which can be compensated through gradient accumulation.

We generate buckets by applying the following algorithm:

Set the width to 256.
While the width is less than or equal to 1024:
Find the largest height such that height is less than or equal to 1024 and that width multiplied by height is less than or equal to 512 * 768.
Add the resolution given by height and width as a bucket.
Increase the width by 64.
The same is repeated with width and height exchanged. Duplicated buckets are pruned from the list, and an additional bucket sized 512x512 is added.

Next, we assign images to their corresponding buckets. For this purpose, we first store the bucket resolutions in a NumPy array and calculate the aspect ratio of each resolution. For each image in the dataset, we then retrieve its resolution and calculate the aspect ratio. The image aspect ratio is subtracted from the array of bucket aspect ratios, allowing us to efficiently select the closest bucket according to the absolute value of the difference between aspect ratios:

image_bucket = argmin(abs(bucket_aspects — image_aspect))
The image's bucket number is stored associated with its item ID in the dataset. If the image's aspect ratio is very extreme and too different from even the best-fitting bucket, the image is pruned from the dataset.

Since we train on multiple GPUs, before each epoch, we shard the dataset to ensure that each GPU works on a distinct subset of equal size. To do this, we first copy the list of item IDs in the dataset and shuffle them. If this copied list is not divisible by the number of GPUs multiplied by the batch size, the list is trimmed, and the last items are dropped to make it divisible.

We then select a distinct subset of 1/world_size*bsz item IDs according to the global rank of the current process. The rest of the custom batch generation will be described as seen from any single of these processes and operate on the subset of dataset item IDs.

For the current shard, lists for each bucket are created by iterating over the list of shuffled dataset item IDs and assigning the ID to the list corresponding to the bucket that was assigned to the image.

Once all images are processed, we iterate over the lists for each bucket. If its length is not divisible by the batch size, we remove the last elements on the list as necessary to make it divisible and add them to a separate catch-all bucket. As the overall shard size is guaranteed to contain a number of elements divisible by the batch size, doing is guaranteed to produce a catch-all bucket with a length divisible by the batch size as well.

When a batch is requested, we randomly draw a bucket from a weighted distribution. The bucket weights are set as the size of the bucket divided by the size of all remaining buckets. This ensures that even with buckets of widely varying sizes, the custom batch generation does not introduce strong bias when during training, an image shows up according to image size. If buckets were chosen without weighting, small buckets would empty out early during the training process, and only the biggest buckets would remain towards the end of training. Weighting buckets by size avoids this.

A batch of items is finally taken from the chosen bucket. The items taken are removed from the bucket. If the bucket is now empty, it is deleted for the rest of the epoch. The chosen item IDs and the chosen bucket's resolution are now passed to an image-loading function.

Image Loading
Note that image loading code is not part of this release but should be relatively easy to implement.

Each item ID's image is loaded and processed to fit within the bucket resolution. For fitting the image, two approaches are possible.

First, the image could be simply rescaled. This would lead to a slight distortion of the image. For this reason, we have opted for the second approach:

The image is scaled, while preserving its aspect ratio, in such a way that it:

Either fits the bucket resolution exactly if the aspect ratio happens to match
or it extends past the bucket resolution on one dimension while fitting it exactly on the other.
In the latter case, a random crop is applied.

As we found that the mean aspect ratio error per image is only 0.033, these random crops only remove very little of the actual image, usually less than 32 pixels.

The loaded and processed images are finally returned as the image part of the batch.
"""
def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, max_size=(512,512), divisible=16, step_size=8, min_dim=256, base_res=(512,512), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
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

    def load_res_map(self, bucket_file, valid_ids=None):
        with open(bucket_file, "rb") as fh:
            self.res_map = pickle.load(fh)
        if valid_ids is not None:
            new_res_map = {}
            valid_ids = set(valid_ids)
            for k, v in self.res_map.items():
                if k in valid_ids:
                    new_res_map[k] = v
            self.res_map = new_res_map

        self.assign_buckets()
        self.start_epoch()

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

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            print(f"skipped images: {skipped}")
            print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
            for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                print(f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
            print(f"assign_buckets: {timer:.5f}s")

    def get_ideal_resolution(self, image_size) -> tuple[int, int]:
        w, h = image_size
        aspect = float(w)/float(h)
        bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
        return self.resolutions[bucket_id]

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = np.array(sorted(list(self.res_map.keys())))
        index_len = index.shape[0]
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        #print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = index.shape[0] // self.bsz
        assert(index.shape[0] % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index], dtype=np.int64)
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):# -> tuple[Any | list | None, Any | tuple[Any, ...]]:
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert(found_batch or len(self.left_over) >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()

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
    res_prompts = {}
    for resolution, batch in queue.items():
        ks = batch.keys()
        imgs = list(batch.values())

        prompts = moondream_model.batch_answer(
            images=imgs,
            prompts=["Caption this image."] * len(imgs),
            tokenizer=moondream_tokenizer,
        )
        prompts_list = [prompt for prompt in prompts]
        for key, prompt in zip(ks, prompts_list):
            res_prompts[key] = prompt

        imgs = [preprocess_image(image) for image in imgs]

        # Convert list of imgs to tensor of batch images
        imgs = torch.stack(imgs, dim=0).to(device)
        latents = vae.encode(imgs).latent_dist.sample().cpu()
        latents_list = [latent for latent in latents]
        for key, latent in zip(ks, latents_list):
            res_latents[key] = latent

    # Reorder latents based on keys
    latents = [res_latents[key] for key in keys]
    prompts = [res_prompts[key] for key in keys]

    texts = siglip_tokenizer(prompts, context_length=siglip_model.context_length).to(device)
    text_embeddings = siglip_model.encode_text(texts)

    return latents, text_embeddings.cpu(), prompts

def create_schema():
    """Create schemas for latents and embeddings parquet tables."""
    latents_schema = pa.schema([
        ('image_id', pa.string()),
        ('latent', pa.list_(pa.float32())),
        ('latent_shape', pa.list_(pa.int64())),
        ('embedding', pa.list_(pa.float32())),
        ('prompt', pa.string())
    ])
    return latents_schema

def write_parquet(latents_list, latents_parquet_file, latents_schema):
    """Write latents and embeddings data into parquet files."""
    latents_table = pa.Table.from_pydict({
        'image_id': [item['image_id'] for item in latents_list],
        'latent': [item['latent'] for item in latents_list],
        'latent_shape': [item['latent_shape'] for item in latents_list],
        'embedding': [item['embedding'] for item in latents_list],
        'prompt': [item['prompt'] for item in latents_list]
    }, schema=latents_schema)

    pq.write_table(latents_table, latents_parquet_file)

def upload_and_delete_files(api, latents_parquet_file, rank, index):
    print(f"Uploading parquet for rank {rank} and index {index}")
    # Upload files
    api.upload_file(
        path_or_fileobj=latents_parquet_file,
        path_in_repo=f"{rank}/latents/{index}_latents.parquet",
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
    )

    # Delete local files
    os.remove(latents_parquet_file)

def main(rank: int, world_size: int, dataset, vae, siglip_model, tokenizer, bucket_manager, api, moondream_model, moondream_tokenizer):
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

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        progress_bar = tqdm(desc=f"Approximate processed", unit="img") if rank == 0 else None
        for i, batch in enumerate(dataset):
            # Calculate latents and embeddings
            latents, text_embeddings, prompts = calculate_latents_and_embeddings(batch, vae, siglip_model, tokenizer, device, bucket_manager, moondream_model, moondream_tokenizer)
            image_ids = batch[IMAGE_ID_COLUMN_NAME]

            for image_id, latent, text_embedding, prompt in zip(image_ids, latents, text_embeddings, prompts):
                image_id_res_map[image_id] = latent.shape[1:]
                image_id_caption_map[image_id] = prompt

                # Append to the lists
                latents_list.append({
                    'image_id': str(image_id),
                    'latent': latent.numpy().flatten().tolist(),
                    'latent_shape': list(latent.shape),
                    'embedding': text_embedding.numpy().flatten().tolist(),
                    'prompt': prompt
                })

            # Write Parquet files after reaching the IMAGES_PER_PARQUET threshold
            if (i + 1) % IMAGES_PER_PARQUET//BATCH_SIZE_PER_GPU == 0:
                index = i // (IMAGES_PER_PARQUET//BATCH_SIZE_PER_GPU)
                latents_parquet_file = f"{latents_dir}/{index}_latents.parquet"

                write_parquet(latents_list, latents_parquet_file, latents_schema)

                # Submit the upload and delete task to the executor
                executor.submit(upload_and_delete_files, api, latents_parquet_file, rank, index)

                # Clear the lists for the next batch
                latents_list.clear()

            progress_bar.update(BATCH_SIZE_PER_GPU * world_size)

        # Handle remaining data after loop ends
        if latents_list:
            index = (i // (IMAGES_PER_PARQUET//BATCH_SIZE_PER_GPU)) + 1
            latents_parquet_file = f"{latents_dir}/{index}_latents.parquet"

            write_parquet(latents_list, latents_parquet_file, latents_schema)

            # Submit the upload and delete task to the executor
            executor.submit(upload_and_delete_files, api, latents_parquet_file, rank, index)

    # Wait for all uploads to complete
    executor.shutdown(wait=True)

    resmap_dir = f"{CACHE_DIR_BASE}/datasets/{DATASET_NAME}/resmaps"
    caption_dir = f"{CACHE_DIR_BASE}/datasets/{DATASET_NAME}/captions"
    os.makedirs(resmap_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)

    # Save res map
    with open(f"{resmap_dir}/{rank}_res_map.json", "w+") as fh:
        json.dump(image_id_res_map, fh)

    # Save caption map
    with open(f"{caption_dir}/{rank}_caption_map.json", "w+") as fh:
        json.dump(image_id_caption_map, fh)

    # Upload res and caption maps
    api.upload_file(
        path_or_fileobj=f"{CACHE_DIR_BASE}/datasets/{DATASET_NAME}/resmaps/{rank}_res_map.json",
        path_in_repo=f"{rank}/res_map.json",
        repo_id=f"{USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=f"{CACHE_DIR_BASE}/datasets/{DATASET_NAME}/captions/{rank}_caption_map.json",
        path_in_repo=f"{rank}/captions.json",
        repo_id=f"{USERNAME}/commoncatalog_cc_by_moondream_captions",
        repo_type="dataset",
    )

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    # datasets.config.HF_HUB_OFFLINE = 1 # Comment this out if you havent downloaded the dataset yet
    # dataset = load_dataset("sroecker/recap-coco30k-moondream", split="train", cache_dir=f"{CACHE_DIR_BASE}/datasets/coco30k_moondream")
    dataset = load_dataset("common-canvas/commoncatalog-cc-by", split="train", streaming=True, columns=["key", "jpg"]).take(1000)
    # moondream_captions_dataset = load_dataset("SwayStar123/commoncatalog-cc-by-moondream_captions", split="train")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir=f"{CACHE_DIR_BASE}/models/vae")
    siglip_model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=f"{CACHE_DIR_BASE}/models/siglip")
    siglip_tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384', cache_dir=f"{CACHE_DIR_BASE}/models/siglip")

    bucket_manager = BucketManager()
    api = HfApi()
    api.create_repo(repo_id=f"{USERNAME}/{DATASET_NAME}", repo_type="dataset", exist_ok=True)
    api.create_repo(repo_id=f"{USERNAME}/commoncatalog_cc_by_moondream_captions", repo_type="dataset", exist_ok=True)

    model_id = "vikhyatk/moondream2"
    revision = "2024-07-23"
    moondream_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, cache_dir=f"{CACHE_DIR_BASE}/models/moondream"
    )
    moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=f"{CACHE_DIR_BASE}/models/moondream")

    mp.spawn(main, args=(world_size, dataset, vae, siglip_model, siglip_tokenizer, bucket_manager, api, moondream_model, moondream_tokenizer), nprocs=world_size)
