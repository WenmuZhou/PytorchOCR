from torchvision.transforms import ColorJitter as _ColorJitter

__all__  = ['ColorJitter']

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,**kwargs):
        self.aug = _ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        image = data['image']
        image = self.aug(image)
        data['image'] = image
        return data
