from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import torch
from utils import norm_img, denorm_img, tight_crop_image, add_padding, centering_image

# draw
def draw_single_char(ch, font, canvas_size):
    image = Image.new('L', (canvas_size, canvas_size), color=255)
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(ch, font=font)

    drawing.text(
        ((canvas_size-w)/2, (canvas_size-h)/2),
        ch,
        fill=(0),
        font=font
    )
    flag = np.sum(np.array(image))

    if flag == 255 * 128 * 128:
        print('No font :', ch)
        return None

    if w>canvas_size or h>canvas_size:
        print('Verify Size')

    return image


# font to pickle
def ttf_to_pkl(text, font_path, pkl_path, canvas_size=128, font_size=90):
    font = ImageFont.truetype(font=font_path, size=font_size)
    text_img = []

    for i in text:
        img = draw_single_char(i, font, canvas_size)
        text_img.append(np.array(img))

    with open(pkl_path, 'wb') as f:
        pickle.dump(text_img, f)


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def init_embedding(PATH, cg_num, char_num):
  c_vec = np.random.normal(size=(cg_num, 128, 1, 1), scale=0.9)
  torch.save(c_vec.reshape(cg_num, 1, 128), PATH + '/save/categorical_vector.pth')
  print("[Success] Saved 'categorical vector' in" + PATH + '/save')

  c_vec_t = torch.FloatTensor([c_vec[i] for i in range(cg_num) for _ in range(char_num)])
  c_vec_t = c_vec_t.cuda()

  return c_vec_t


def center_align(source, target):
    re_source = []
    for i in source:
      cen = centering_image(i, image_size =128)
      re_source.append(cen)

    re_target = []
    for i in target:
      tmp = []
      for j in i:
        cen = centering_image(j, image_size=128)
        tmp.append(cen)
      re_target.append(tmp)

    source = np.array(re_source)
    target = np.array(re_target)
    return source, target


def get_text():
    upper = "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z"
    lower = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z"
    number = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"

    text = (upper+', '+lower+', '+number).split(", ")
    return text


def load_train_data(PATH, num_of_style = 5):
    text = get_text()
    files = os.listdir(PATH + '/font')

    ttf_to_pkl(text, PATH +'/font/' + 'source_font.ttf', 'source_font.pkl', canvas_size=128)
    source = load_pkl('source_font.pkl')
    source = np.array(source)

    t_array = []
    for (num, f) in enumerate(files):
      if num == num_of_style:
        break
      font_path = os.path.join(PATH + '/font', f)

      filename, _ = os.path.splitext(f)
      pkl_path = filename + '.pkl'
      ttf_to_pkl(text, font_path, pkl_path, canvas_size = 128)

      t_array.append(load_pkl(pkl_path))

    target = np.array(t_array)
    source, target = center_align(source, target)

    # making tensor
    source_tensor = torch.FloatTensor(source)
    target_tensor = torch.FloatTensor(target)

    source_tensor = norm_img(source_tensor)
    target_tensor = norm_img(target_tensor)

    cg_num, font_num = target_tensor.shape[0], target_tensor.shape[1]

    torch.save(source_tensor, PATH + '/save/source_tensor.pth')
    torch.save(target_tensor, PATH + '/save/target_tensor.pth')
    print("[Success] Saved 'source tensor' and 'target tensor' in" + PATH + '/save')

    c_vec_t = init_embedding(PATH, cg_num, font_num)
    print('[Success]: loaded train data')

    return source_tensor, target_tensor, c_vec_t
