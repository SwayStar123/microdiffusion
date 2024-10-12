import torch
import torch.nn.functional as F

def apply_mask_to_tensor(x, mask, patch_size):
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (torch.Tensor): Tensor of shape (bs, c, h, w)
        mask (torch.Tensor): Tensor of shape (bs, num_patches)
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
    bs, c, h, w = x.shape
    num_patches_h = h // patch_size[0]
    num_patches_w = w // patch_size[1]

    # Ensure that height and width are divisible by patch_size
    assert h % patch_size[0] == 0 and w % patch_size[1] == 0, "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(h, w, patch_size)

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.view(bs, num_patches_h, num_patches_w)

    # Expand the mask to cover each patch
    # (bs, num_patches_h, num_patches_w) -> (bs, 1, h, w)
    mask = mask.unsqueeze(1)  # Add channel dimension
    mask = mask.repeat(1, 1, patch_size[0], patch_size[1])  # Repeat for patch_size
    mask = mask.view(bs, 1, h, w)  # Reshape to (bs, 1, h, w)

    # Apply the mask to the input tensor
    x = x * mask

    return x

def unpatchify(x, patch_size, height, width):
    """
    Reconstructs images from patches.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_patches, patch_size * patch_size * in_channels)
        patch_size (int): Size of each patch.
        height (int): Original image height.
        width (int): Original image width.

    Returns:
        torch.Tensor: Reconstructed image of shape (bs, in_channels, height, width)
    """
    bs, num_patches, patch_dim = x.shape
    H, W = patch_size
    in_channels = patch_dim // (H * W)

    # Calculate the number of patches along each dimension
    num_patches_h = height // H
    num_patches_w = width // W

    # Ensure num_patches equals num_patches_h * num_patches_w
    assert num_patches == num_patches_h * num_patches_w, "Mismatch in number of patches."

    # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
    x = x.view(bs, num_patches_h, num_patches_w, H, W, in_channels)

    # Permute x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape x to (bs, height, width, in_channels)
    reconstructed = x.view(bs, height, width, in_channels)

    # Permute back to (bs, in_channels, height, width)
    reconstructed = reconstructed.permute(0, 3, 1, 2).contiguous()

    return reconstructed

def strings_to_tensor(string_list):
    """
    Converts a list of strings, each representing a list (e.g., "[1, 2, 3]"), 
    into a PyTorch tensor.

    Args:
        string_list (list of str): A list of strings, where each string is a list in string form.

    Returns:
        torch.Tensor: A PyTorch tensor containing the data from the lists.
    """
    # Convert each string to a list using eval
    list_of_lists = [eval(s) for s in string_list]
    
    # Convert the list of lists to a PyTorch tensor
    tensor = torch.tensor(list_of_lists, dtype=torch.float32)
    
    return tensor

def random_mask(bs: int, height: int, width: int, patch_size: tuple[int, int], mask_ratio: float) -> torch.Tensor:
    """
    Generates a random mask for patched images. Randomly selects patches to mask.

    Args:
        bs (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        patch_size (tuple of int): Size of the patches.
        mask_ratio (float): Ratio of patches to mask. Ranges from 0 to 1. mask_ratio * 100 = percentage of 1s in the mask

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_patches) with values in {0, 1}.
    """
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)
    
    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_patches)
    
    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)
    
    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_patches)
    
    # Set the first num_patches_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_patches_to_mask]] = 0
    
    # Ensure the final shape is (bs, num_patches)
    mask = mask.view(bs, num_patches)

    return mask

def remove_masked_patches(patches, mask):
    """
    Removes the masked patches from the patches tensor while preserving batch dimensions.
    Returned tensor will have shape (bs, number_of_unmasked_patches, embed_dim).
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()
    mask = mask.logical_not()

    # Get batch size and embed dimension
    bs, num_patches, embed_dim = patches.shape

    # Expand mask to match the shape of patches for correct indexing
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    # Use masked_select and reshape to maintain batch size
    unmasked_patches = torch.masked_select(patches, ~mask).view(bs, -1, embed_dim)

    return unmasked_patches

def add_masked_patches(patches, mask):
    """
    Adds the masked patches to the patches tensor.
    Returned tensor will have shape (bs, num_patches, embed_dim).
    The missing patches will be filled with 0s.
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Get the total number of patches and embed dimension
    bs, num_patches, embed_dim = mask.shape[0], mask.shape[1], patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = torch.zeros(bs, num_patches, embed_dim, device=patches.device, dtype=patches.dtype)

    # Iterate over each batch and place unmasked patches back in their original positions
    for i in range(bs):
        # Use the mask to place unmasked patches back in the correct positions
        full_patches[i, mask[i]] = patches[i].to(full_patches.dtype)

    return full_patches