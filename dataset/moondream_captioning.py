from datasets import load_dataset
import time
import fal_client
from dotenv import load_dotenv
import requests
import concurrent
import asyncio
import json
from tqdm import tqdm
from aiolimiter import AsyncLimiter

# Initialize the rate limiter: 10 requests per second
rate_limiter = AsyncLimiter(max_rate=10, time_period=1)

# Shared data structures
captions = {}
link_rot_url_keys = []

async def subscribe(inputs):
    """
    Subscribes to the fal.ai API asynchronously to get captions for images.

    :param inputs: A list of dictionaries with 'prompt' and 'image_url'.
    :return: The result from the API call.
    """
    async with rate_limiter:
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                if update.logs:
                    for log in update.logs:
                        print(log["message"])

        result = await fal_client.subscribe_async(
            "fal-ai/moondream/batched",
            arguments={
                "inputs": inputs
            },
            on_queue_update=on_queue_update,
        )

    return result

def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code indicates success
        if response.status_code == 200:
            # Check if the content-type indicates an image
            content_type = response.headers.get('content-type')
            if 'image' in content_type:
                return True
        return False
    except requests.RequestException:
        return False

def clean_urls(batch, max_workers=64):
    """
    Since there may be link rot, this function confirms the links are still valid.
    If the link is valid, the URL is returned in the dictionary, otherwise it is returned in the list.

    :param batch: A list of dictionaries containing the key and URL of the image.
    :param max_workers: The maximum number of threads to use for parallel checking.
    :return: A dictionary containing the valid URLs and a list containing the invalid URLs' keys.
    """
    valid_urls = {}
    invalid_urls_keys = []

    keys = batch["key"]
    urls = batch["downloadurl"]

    def check_url(key, url):
        """Helper function to check a URL and return key, validity status."""
        if is_valid_url(url):
            return key, url, True
        return key, url, False

    # Use ThreadPoolExecutor to parallelize URL checks, specify the number of workers with max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_url, key, url) for key, url in zip(keys, urls)]
        for future in concurrent.futures.as_completed(futures):
            key, url, is_valid = future.result()
            if is_valid:
                valid_urls[key] = url
            else:
                invalid_urls_keys.append(key)

    return valid_urls, invalid_urls_keys


async def process_batch(batch):
    """
    Processes a single batch: cleans URLs, calls the API, and collects captions.

    :param batch: A dictionary containing 'key' and 'downloadurl' lists.
    """
    # Clean URLs
    valid_urls, invalid_urls_keys = clean_urls(batch)
    link_rot_url_keys.extend(invalid_urls_keys)

    # Process valid URLs
    if not valid_urls:
        print("No valid URLs in this batch.")
        return

    # Prepare inputs for the API call
    keys = list(valid_urls.keys())
    urls = list(valid_urls.values())
    inputs = [
        {
            "prompt": "Caption this image.",
            "image_url": url
        } for url in urls
    ]

    # Call the API
    result = await subscribe(inputs)

    # Collect captions
    captions_list = result.get('outputs', [])
    for key, caption in zip(keys, captions_list):
        captions[key] = caption

async def main(batch_size=64, num_concurrent_calls=8):
    load_dotenv()

    # Load the dataset
    dataset = load_dataset(
        "common-canvas/commoncatalog-cc-by",
        split="train",
        streaming=True,
        columns=["key", "downloadurl"]
    ).take(10000)

    batched_dataset = dataset.batch(batch_size=batch_size*num_concurrent_calls)

    progress_bar = tqdm(desc="Processing batches", total=10000)

    for batch in batched_dataset:
        # Split batch into num_concurrent_calls batches
        mini_batches = [
            {
                'key': batch['key'][i:i + batch_size],
                'downloadurl': batch['downloadurl'][i:i + batch_size]
            }
            for i in range(0, len(batch['key']), batch_size)
        ]

        # Process 8 batches concurrently
        await process_batches(mini_batches)

        progress_bar.update(batch_size*num_concurrent_calls)

    # Save the results
    with open('captions.json', 'w') as f:
        json.dump(captions, f)
    with open('link_rot_url_keys.json', 'w') as f:
        json.dump(link_rot_url_keys, f)

async def process_batches(batch_list):
    """
    Processes a list of batches concurrently.

    :param batch_list: A list of batches to process.
    """
    tasks = [process_batch(batch) for batch in batch_list]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())