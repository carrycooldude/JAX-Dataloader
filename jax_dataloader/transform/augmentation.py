import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, List
import random
from jax import random
from functools import partial

@jax.jit
def random_flip(x: jnp.ndarray, key: jnp.ndarray, p: float = 0.5) -> jnp.ndarray:
    """Randomly flip the image horizontally"""
    should_flip = random.bernoulli(key, p)
    return jnp.where(should_flip, jnp.fliplr(x), x)

@jax.jit
def random_rotation(x: jnp.ndarray, key: jnp.ndarray, max_angle: float = 15.0) -> jnp.ndarray:
    """Randomly rotate the image"""
    # For non-image data, just return the input
    if len(x.shape) < 2:
        return x
        
    angle = random.uniform(key, minval=-max_angle, maxval=max_angle) * jnp.pi / 180.0
    cos_theta = jnp.cos(angle)
    sin_theta = jnp.sin(angle)
    
    # For non-image data (e.g. feature vectors), apply a simple rotation
    if len(x.shape) == 2:
        rotation_matrix = jnp.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])
        # Reshape to apply rotation to pairs of features
        n_pairs = x.shape[1] // 2
        x_reshaped = x[:, :2*n_pairs].reshape(-1, n_pairs, 2)
        x_rotated = jnp.einsum('ijk,kl->ijl', x_reshaped, rotation_matrix)
        x_final = x_rotated.reshape(x.shape[0], -1)
        # Keep any remaining features unchanged
        if x.shape[1] > 2*n_pairs:
            x_final = jnp.concatenate([x_final, x[:, 2*n_pairs:]], axis=1)
        return x_final
    
    # For image data (3+ dimensions), use affine transform
    else:
        rotation_matrix = jnp.array([[cos_theta, -sin_theta, 0],
                                   [sin_theta, cos_theta, 0],
                                   [0, 0, 1]])
        return jax.image.transform(x, rotation_matrix, mode='nearest')

@jax.jit
def random_crop(x: jnp.ndarray, key: jnp.ndarray, size: tuple) -> jnp.ndarray:
    """Randomly crop the image"""
    h, w = x.shape[:2]
    h_new, w_new = size
    top = random.randint(key, (), 0, h - h_new)
    left = random.randint(key, (), 0, w - w_new)
    return jax.lax.dynamic_slice(x, (top, left) + (0,) * (x.ndim - 2), size + x.shape[2:])

@jax.jit
def color_jitter(x: jnp.ndarray, key: jnp.ndarray, brightness: float = 0.2,
                contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1) -> jnp.ndarray:
    """Apply color jittering"""
    keys = random.split(key, 4)
    x = x + random.uniform(keys[0], x.shape, minval=-brightness, maxval=brightness)
    x = x * random.uniform(keys[1], (), minval=1-contrast, maxval=1+contrast)
    x = jnp.clip(x, 0, 1)
    return x

@jax.jit
def gaussian_noise(x: jnp.ndarray, key: jnp.ndarray, std: float = 0.1) -> jnp.ndarray:
    """Add Gaussian noise"""
    noise = random.normal(key, x.shape) * std
    return jnp.clip(x + noise, 0, 1)

@jax.jit
def cutout(x: jnp.ndarray, key: jnp.ndarray, size: int = 16) -> jnp.ndarray:
    """Apply cutout augmentation"""
    h, w = x.shape[:2]
    keys = random.split(key, 2)
    center_y = random.randint(keys[0], (), 0, h)
    center_x = random.randint(keys[1], (), 0, w)
    y1 = jnp.clip(center_y - size // 2, 0, h)
    y2 = jnp.clip(center_y + size // 2, 0, h)
    x1 = jnp.clip(center_x - size // 2, 0, w)
    x2 = jnp.clip(center_x + size // 2, 0, w)
    mask = jnp.ones_like(x)
    mask = mask.at[y1:y2, x1:x2].set(0)
    return x * mask

class JAXDataAugmentation:
    def __init__(self, augmentations=None, seed=42):
        self.augmentations = augmentations or ['random_flip', 'random_rotation', 'color_jitter']
        self.rng = jax.random.PRNGKey(seed)
        
    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply random augmentations"""
        self.rng, *subkeys = random.split(self.rng, len(self.augmentations) + 1)
        
        for aug, key in zip(self.augmentations, subkeys):
            if random.bernoulli(key):  # 50% chance to apply each augmentation
                if aug == 'random_flip':
                    x = random_flip(x, key)
                elif aug == 'random_rotation':
                    x = random_rotation(x, key)
                elif aug == 'random_crop':
                    x = random_crop(x, key, (x.shape[0]//2, x.shape[1]//2))
                elif aug == 'color_jitter':
                    x = color_jitter(x, key)
                elif aug == 'gaussian_noise':
                    x = gaussian_noise(x, key)
                elif aug == 'cutout':
                    x = cutout(x, key)
        return x 