import random

import coolname


def generate_slug_with_seed(seed=None, length=3):
    if seed is not None:
        random.seed(seed)
    slug = coolname.generate_slug(length)
    if seed is not None:
        random.seed()
    return slug
