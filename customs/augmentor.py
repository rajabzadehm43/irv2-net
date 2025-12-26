import albumentations as a
from numpy import ndarray as image
import cv2

class Augmentor:
    
    def augment(self, image:image, label:image):
        
        h, w, *_ = image.shape
        
        transform = a.Compose([
            a.RandomSizedCrop((h // 2, h), (h, w)),
            a.Rotate((-30, 30), cv2.INTER_CUBIC),
            a.GridDistortion(),
            a.OpticalDistortion((-0.1, 0.1), cv2.INTER_CUBIC),
            a.VerticalFlip(),
            a.HorizontalFlip(),
            a.RandomBrightnessContrast(contrast_limit=(0, 0)), # random brightness
            a.RandomBrightnessContrast(brightness_limit=(0, 0)), # random contrast
            a.RandomGamma(),
            a.HueSaturationValue(),
            a.ChannelShuffle(),
            a.MotionBlur(),
            a.MedianBlur(),
            a.GaussianBlur(sigma_limit=1.0),
            a.GaussNoise((0.02, 0.1), p=.3),
            a.CoarseDropout((1, 5), p=0.2),
        ], p=.7)
        
        result = transform(image=image, mask=label)
        return result['image'], result['mask']