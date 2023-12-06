import numpy as np

def norm_img(img):
    return img / (255/2) - 1

def denorm_img(img):
    return ((img + 1) / 2).clamp(0, 1)

def tight_crop_image(img, resize_fix=False):
    x1 = img.shape[0]
    x2 = 0
    y1 = img.shape[0]
    y2 = 0
    index = 0
    for i in img:
      tmp = np.array(np.where(i != 255))
      tmp = tmp[0]
      if len(tmp) != 0:
        if index <= y1:
          y1 = index
        if index >= y2:
          y2 = index
        if tmp[np.argmin(tmp)] < x1:
          x1 = tmp[np.argmin(tmp)]
        if tmp[np.argmax(tmp)] > x2:
          x2 = tmp[np.argmax(tmp)]
      index += 1

    cropped_image = img[y1:y2+1, x1:x2+1]
    return cropped_image


def add_padding(img, image_size=128, pad_value=None):
    height, width = img.shape
    if not pad_value:
        pad_value = img[0][0]

    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)

    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)

    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)

    return img


def centering_image(img, image_size=128, resize_fix=False, pad_value=None):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, pad_value=pad_value)

    return centered_image