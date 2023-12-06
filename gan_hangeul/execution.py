import torch
import cv2
from scipy.stats import norm
import matplotlib.pyplot as plt

from utils import denorm_img


def loadData(PATH):
    PATH = PATH + '/save'
    source_tensor = torch.load(PATH + '/source_tensor.pth')
    target_tensor = torch.load(PATH + '/target_tensor.pth')
    c_vector = torch.load(PATH + '/categorical_vector.pth')
    return source_tensor, target_tensor, c_vector

def loadModel(PATH, epoch):
    PATH = PATH + '/save/'

    encoder = torch.load(PATH+'encoder-'+str(epoch)+'.pth')
    decoder = torch.load(PATH+'decoder-'+str(epoch)+'.pth')
    discriminator = torch.load(PATH+'discriminator-'+str(epoch)+'.pth')

    encoder.eval()
    decoder.eval()
    discriminator.eval()

    return encoder, decoder, discriminator


def processed_image(img, kernel_size= 5, kernel_shape= 'ellipse', iteration = 1):
    if kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        print('[warning] wrong \'kernel_shape\' input. Try with \'ellipse\', \'rect\' or \'cross\'.')
    result_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations= 1)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, kernel, iterations= 1)
    result_img = torch.from_numpy(result_img)
    return result_img


def generate(encoder, decoder, discriminator, source_tensor, c_vec, ch, style):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    x = torch.FloatTensor(source_tensor[ch]).view((1, 1, 128, 128)).cuda()
    c = torch.FloatTensor(c_vec[style]).view((1, 128, 1, 1)).cuda()

    z, e_net = encoder(x)
    d_in = torch.cat((z, c), dim=1)
    d_out = decoder(d_in, e_net).detach()
    image = d_out.cpu()
    denorm_image = denorm_img(image)
    return image, denorm_image


def custom_generate(encoder, decoder, discriminator, source_tensor, c_vec, ch):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    x = torch.FloatTensor(source_tensor[ch]).view((1, 1, 128, 128)).cuda()
    c = torch.FloatTensor(c_vec).view((1, 128, 1, 1)).cuda()
    z, e_net = encoder(x)
    d_in = torch.cat((z, c), dim=1)
    d_out = decoder(d_in, e_net).detach()
    image = d_out.cpu()
    denorm_image = denorm_img(image)
    return image, denorm_image


def sub_distance(a, b, s):
    return a + (b - a) * s


def sample_normal_distribution(num_of_point = 100):
    rv = norm(loc = 0, scale = 1)
    n = num_of_point+1

    end_point = 1 - 1/n
    start_point = 1/n
    width = rv.ppf(end_point) - rv.ppf(start_point)
    zero_point = rv.ppf(end_point)

    normDisInvList = []
    for i in range(1, n):
        normDisInvList.append((rv.ppf(i/n)+zero_point)/width*num_of_point)

    return normDisInvList


# for visualizing c-vector
def print_scroll_vec(a, b, n=10, norm=False):
    if norm:
        pointlist = sample_normal_distribution(n)
    else:
        pointlist = range(n)

    for point in pointlist:
        sp = sub_distance(a, b, point).numpy()[0]
        plt.scatter(sp[0], sp[1])
    plt.show()


# print scrolling image
def print_scroll_img(encoder, decoder, discriminator, source_tenser, a, b, ch=0, n=10, norm = False, imshow=False, process = False):
    arr = []
    if norm:
        pointlist = sample_normal_distribution(n)
    else:
        pointlist = range(n)

    for i in pointlist:
        sp = sub_distance(a, b, i/(n-1)).numpy()[0]
        image, denorm_image = custom_generate(encoder, decoder, discriminator, source_tensor, sp.reshape(1, 128, 1, 1), ch)
        denorm_image = denorm_image[0,0]
        if process:
            denorm_image = processed_image(denorm_image.numpy(), kernel_size = 5)
        if imshow:
            plt.imshow(denorm_image, cmap='gray')
            plt.show()
        arr.append(denorm_image)
    return arr

