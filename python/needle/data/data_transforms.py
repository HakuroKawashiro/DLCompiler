import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        rev_img = img
        if flip_img:
            rev_img = np.flip(img, axis = 1)
        
        return rev_img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        
        padding_img = np.pad(img,
                             pad_width=((self.padding, self.padding),
                                        (self.padding, self.padding),
                                        (0, 0)),
                             mode='constant',
                             constant_values=0)
        
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y
        
        cropped_img = padding_img[start_x: start_x + H, start_y: start_y + W, :]
        
        return cropped_img
        ### END YOUR SOLUTION
