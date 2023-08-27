from PIL import Image
import numpy as np
from torchvision.transforms import RandAugment as RawRandAugment

class RandAugment(RawRandAugment):
    def __init__(self, prob=0.5, *args, **kwargs):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        img = data['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)
        data['image'] = img
        return data
