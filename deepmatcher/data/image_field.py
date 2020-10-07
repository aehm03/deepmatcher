# TODO more than one Image
# TODO unifiying class for text and image fields?
import torch


class ImageField:
    """
    Represents a field containing an image

    """

    def __init__(self):
        pass

    # TODO: load from disk? how does deepmatcher do batches? -> MatchingIterator

    def get_vector_data(self):
        """
        just return a dummy vector
        """
        return torch.Tensor([[1.0, 0, 0], [0.0, 0, 1.0]])
