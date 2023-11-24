import numpy as np

def crop_right_tool(frame: np.ndarray) -> np.ndarray:
    """
    Crops the right tool from the frame
    :param frame: The frame to crop
    :return: The cropped frame
    """
    width = frame.shape[1]

    return frame[:, width//2:, :]

def crop_frame(frame: np.ndarray, crop_size: int = 256, center: bool = True):
    """
    Crops the image to the desired size
    :param frame: The image to crop
    :param crop_size: The desired size
    :param center: Whether to crop the image from the center or from the top left corner
    :return: The cropped image
    """
    if center:
        h, w = frame.shape[:2]
        x = int((w - crop_size) / 2)
        y = int((h - crop_size) / 2)
        return frame[y:y + crop_size, x:x + crop_size]
    else:
        return frame[:crop_size, :crop_size]