# Released under MIT license
# Copyright (c) 2022 finetuneanon (NovelAI/Anlatan LLC)
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

Sample Output
resolutions:
[[ 256 1024] [ 320 1024] [ 384 1024] [ 384  960] [ 384  896] [ 448  832] [ 512  768] [ 512  704]
 [ 512  512] [ 576  640] [ 640  576] [ 704  512] [ 768  512] [ 832  448] [ 896  384] [ 960  384]
 [1024  384] [1024  320] [1024  256]]
aspects:
[0.25       0.3125     0.375      0.4        0.42857143 0.53846154
 0.66666667 0.72727273 1.         0.9        1.11111111 1.375
 1.5        1.85714286 2.33333333 2.5        2.66666667 3.2
 4.        ]
gen_buckets: 0.00012s
skipped images: 344
aspect error: mean 0.03291049121627481, median 0.022727272727272707, max 3.991769547325103
bucket 7: [512 704], aspect 0.72727, entries 2189784
bucket 6: [512 768], aspect 0.66667, entries 661814
bucket 11: [704 512], aspect 1.37500, entries 586743
bucket 8: [512 512], aspect 1.00000, entries 429811
bucket 9: [576 640], aspect 0.90000, entries 417592
bucket 5: [448 832], aspect 0.53846, entries 247869
bucket 10: [640 576], aspect 1.11111, entries 231463
bucket 12: [768 512], aspect 1.50000, entries 209557
bucket 13: [832 448], aspect 1.85714, entries 173294
bucket 4: [384 896], aspect 0.42857, entries 42191
bucket 1: [ 320 1024], aspect 0.31250, entries 38428
bucket 2: [ 384 1024], aspect 0.37500, entries 24377
bucket 0: [ 256 1024], aspect 0.25000, entries 16618
bucket 14: [896 384], aspect 2.33333, entries 15046
bucket 3: [384 960], aspect 0.40000, entries 12010
bucket 16: [1024  384], aspect 2.66667, entries 4512
bucket 17: [1024  320], aspect 3.20000, entries 3704
bucket 15: [960 384], aspect 2.50000, entries 3478
bucket 18: [1024  256], aspect 4.00000, entries 2326
assign_buckets: 15.47429s
"""

import numpy as np
import pickle
import time
import json

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, bucket_file, valid_ids=None, max_size=(512,512), divisible=16, step_size=8, min_dim=256, base_res=(512,512), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
        with open(bucket_file, "r") as fh:
            self.res_map = json.load(fh)
        if valid_ids is not None:
            new_res_map = {}
            valid_ids = set(valid_ids)
            for k, v in self.res_map.items():
                if k in valid_ids:
                    new_res_map[k] = v
            self.res_map = new_res_map
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
        self.left_over = {}
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
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
        self.left_over = {}
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index], dtype=np.int64)
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    if bucket_id not in self.left_over:
                        self.left_over[bucket_id] = []
                    self.left_over[bucket_id].extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        if self.epoch is None or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]

            left_over_bucket_ids = list(self.left_over.keys())
            left_over_bucket_probs = [len(self.left_over[bucket_id]) for bucket_id in left_over_bucket_ids]

            all_bucket_ids = bucket_ids + left_over_bucket_ids
            all_bucket_probs = bucket_probs + left_over_bucket_probs

            if len(all_bucket_probs) == 0:
                # No buckets left, start new epoch
                self.start_epoch()
                continue

            all_bucket_probs = np.array(all_bucket_probs, dtype=np.float32)
            all_bucket_probs = all_bucket_probs / all_bucket_probs.sum()
            all_bucket_ids = np.array(all_bucket_ids, dtype=np.int64)

            chosen_id = int(self.prng.choice(all_bucket_ids, 1, p=all_bucket_probs)[0])

            if chosen_id in self.epoch:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # Move leftovers to left_over dict
                    if chosen_id not in self.left_over:
                        self.left_over[chosen_id] = []
                    self.left_over[chosen_id].extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]
            elif chosen_id in self.left_over:
                if len(self.left_over[chosen_id]) >= self.bsz:
                    batch_data = self.left_over[chosen_id][:self.bsz]
                    self.left_over[chosen_id] = self.left_over[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.left_over[chosen_id]) == 0:
                        del self.left_over[chosen_id]
                else:
                    # Not enough images to form a batch, keep them for the next epoch
                    del self.left_over[chosen_id]
            else:
                # Should not happen
                assert False, "Chosen bucket ID not found in epoch or leftovers"

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

if __name__ == "__main__":
    # prepare a pickle with mapping of dataset IDs to resolutions called resolutions.pkl to use this
    with open("resolutions.pkl", "rb") as fh:
        ids = list(pickle.load(fh).keys())

    counts = np.zeros((len(ids),)).astype(np.int64)
    id_map = {}
    for i, post_id in enumerate(ids):
        id_map[post_id] = i

    bm = BucketManager("resolutions.pkl", debug=True, bsz=8, world_size=8, global_rank=3)
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))

    bm = BucketManager("resolutions.pkl", bsz=8, world_size=1, global_rank=0, valid_ids=ids[0:16])
    for _ in range(16):
        bm.get_batch()
    print("got from future epoch: " + str(bm.get_batch()))

    bms = []
    for rank in range(16):
        bm = BucketManager("resolutions.pkl", bsz=8, world_size=16, global_rank=rank)
        bms.append(bm)
    for epoch in range(5):
        print(f"epoch {epoch}")
        for i, bm in enumerate(bms):
            print(f"bm {i}")
            first = True
            for ids, res in bm.generator():
                if first and i == 0:
                    #print(ids)
                    first = False
                for post_id in ids:
                    counts[id_map[post_id]] += 1
        print(np.bincount(counts))