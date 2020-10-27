import os
from typing import List
import torch

from PIL import Image
from deepmatcher.data import MatchingField
from torchvision import transforms


class ImageField(MatchingField):
    """
    Represents a field containing an image
    Did not want to use the Field/RawField classes from torchtext but implements some methods from them.
    Right now the following rationale is hard-coded:
    * all images lie in a single directory (image_path)
    * images are identified with an id, full path to image is: $image_path/$id_0.png
    * not all products have an image, they are replaced with empyt tensors
    """

    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.use_vocab = False
        self.is_target = False
        self.image_transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop((200, 200)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])  # should be standard imagenet normalization

    def process(self, batch: List[object], device=None):
        """ Process a list of examples to create a batch / torch.Tensor.
        """
        # this will also be called when computing the metadata, makes it expensive but what can we do

        images = []

        for image_path in batch:
            if image_path is not None:
                im = Image.open(image_path).convert('RGBA')

                # Transform transparent images
                background = Image.new('RGBA', im.size, (255, 255, 255))
                alpha_composite = Image.alpha_composite(background, im).convert('RGB')
                im_final = self.image_transformation(alpha_composite)
                images.append(im_final)
            else:
                images.append(torch.zeros((3, 200, 200)))

        stacked = torch.stack(images).to(device)

        return stacked

    def preprocess(self, val):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """

        # checks if this product has an image
        # for now we only take the first one
        path = os.path.join(self.image_directory, val + '.png')
        if os.path.exists(path):
            return path
        else:
            return None

    def preprocess_args(self):
        return {'image_directory': self.image_directory}
