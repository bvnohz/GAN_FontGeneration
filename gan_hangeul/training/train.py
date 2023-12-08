import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataload import load_train_data
from models import Encoder, Decoder, Discriminator
from execution import *

def shift_and_resize_image(source_set, target_set):
    batch_size, channel, w, h = source_set.shape
    multiplier = np.random.uniform(1.00, 1.20)
    nw, nh = int(w * multiplier) + 1, int(h * multiplier) + 1

    shift_x = int(np.ceil(np.random.uniform(low= 0.01, high= nw-w)))
    shift_y = int(np.ceil(np.random.uniform(low = 0.01, high = nh-h)))

    src_out = F.interpolate(source_set, size = [nw, nh], mode='bilinear', align_corners=True)
    tar_out = F.interpolate(target_set, size = [nw, nh], mode='bilinear', align_corners=True)

    src_out = src_out[:, :, shift_x : shift_x + w , shift_y : shift_y + h]
    tar_out = tar_out[:, :, shift_x : shift_x + w , shift_y : shift_y + h]

    return src_out, tar_out


def gprint(encoder, decoder, discriminator, source_tensor, target_tensor, c_vec, ch):
    cg_num = target_tensor.shape[0]
    g_list = []
    dg_list = []
    for i in range(cg_num):
        g, dg = generate(encoder, decoder, discriminator, source_tensor, c_vec, ch, i)
        g_list.append(g)
        dg_list.append(dg)
    plt.figure(figsize=(25, 4))
    plt.title('Real')

    for i in range(cg_num):
        plt.subplot(1, cg_num, i+1)
        plt.imshow(target_tensor[i, ch], cmap = 'gray')
    plt.figure(figsize=(25, 4))
    plt.show()

    for i in range(cg_num):
        plt.title('Fake')
        plt.subplot(1, cg_num, i+1)
        plt.imshow(dg_list[i].cpu().numpy()[0, 0], cmap='gray')
    plt.show()


def train(PATH, epochs = 300, batch_size = 16, check_point = 50, learning_rate = 0.0001, num_of_style = 5, verbose = False):
    source_tensor, target_tensor, c_vec_t = load_train_data(PATH, num_of_style = num_of_style)
    cg_num, font_num = target_tensor.shape[0], target_tensor.shape[1]

    x_data = torch.FloatTensor(source_tensor).repeat(cg_num, 1, 1).view((cg_num * font_num, 1, 128, 128))
    t_data = torch.FloatTensor(target_tensor).view((cg_num * font_num, 1, 128, 128))
    t_meta = torch.LongTensor([[j, i] for j in range(cg_num) for i in range(font_num)]) # [cg, font]

    x_data = x_data.cuda()
    t_data = t_data.cuda()
    c_vec = torch.load(PATH + '/save/categorical_vector.pth')

    dataset = TensorDataset(x_data, t_data, t_meta, c_vec_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    discriminator = Discriminator(category_num=cg_num).cuda()

    encoder.train()
    decoder.train()
    discriminator.train()

    l1_criterion = nn.L1Loss().cuda()
    bce_criterion = nn.BCEWithLogitsLoss().cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    mse_criterion = nn.MSELoss().cuda()

    g_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    L1_W = 100
    MSE_W = 15

    print('batch_size: %d, category_num: %d, font_num: %d'%(batch_size, cg_num, font_num))
    log = []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        d_total_loss = []
        g_total_loss = []

        l1_loss_arr = []
        ff_bce_loss_arr = []
        bce_loss_arr = []
        ce_loss_arr = []
        z_loss_arr = []

        for x, t, tm, c in loader:
            x, t = shift_and_resize_image(x, t)
            cg = tm.T[0]
            font = tm.T[0]

            z, e_net = encoder(x)

            d_in = torch.cat((z, c), dim=1)
            d_out = decoder(d_in, e_net)

            real_tf, real_cg = discriminator(t)
            fake_tf, fake_cg = discriminator(d_out)

            one_tensor = torch.ones(x.shape[0], 1).cuda()
            zero_tensor = torch.zeros(x.shape[0], 1).cuda()

            r_bce_loss = bce_criterion(real_tf, one_tensor)
            f_bce_loss = bce_criterion(fake_tf, zero_tensor)
            bce_loss = r_bce_loss + f_bce_loss

            category = torch.FloatTensor(np.array(np.eye(cg_num)[cg])).cuda()
            r_ce_loss = bce_criterion(real_cg, category)
            f_ce_loss = bce_criterion(fake_cg, category)
            ce_loss = 0.5*(r_ce_loss + f_ce_loss)

            t_z = encoder(d_out)[0]
            z_loss = mse_criterion(z, t_z)
            l1_loss = l1_criterion(d_out, t)
            ff_bce_loss = bce_criterion(fake_tf, one_tensor)
            g_loss = (L1_W * l1_loss) + ff_bce_loss + (MSE_W * z_loss) + f_ce_loss
            d_loss = bce_loss + ce_loss

            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)

            encoder.zero_grad()
            decoder.zero_grad()
            g_loss.backward(retain_graph=True)

            g_optimizer.step()
            d_optimizer.step()

            d_total_loss.append(d_loss.item())
            g_total_loss.append(g_loss.item())

            l1_loss_arr.append(l1_loss.item())
            ff_bce_loss_arr.append(ff_bce_loss.item())
            bce_loss_arr.append(bce_loss.item())
            ce_loss_arr.append(ce_loss.item())
            z_loss_arr.append(z_loss.item())
        log.append([d_total_loss, g_total_loss, l1_loss_arr, ff_bce_loss_arr, bce_loss_arr, ce_loss_arr, z_loss_arr])

        if verbose:
            print('epoch: %d/%d\tg_loss: %f\td_loss: %f'%(epoch+1, epochs, sum(g_total_loss), sum(d_total_loss)))
            if (epoch+1) % 10 == 0:
                gprint(source_tensor, target_tensor, c_vec, np.random.randint(0, source_tensor.shape[0]))

        if (epoch+1) % check_point == 0 or (epoch+1) == epochs:
            torch.save(encoder, PATH+'/save/encoder-'+str(epoch+1)+'.pth')
            torch.save(decoder, PATH+'/save/decoder-'+str(epoch+1)+'.pth')
            torch.save(discriminator, PATH+'/save/discriminator-'+str(epoch+1)+'.pth')
            print('epoch: %d save checkpoint'%(epoch+1))


### training start ###
# PATH='.'
# train(PATH=PATH, epochs =300, check_point = 50, learning_rate = 0.0001)
# source_tensor, target_tensor, c_vec = loadData(PATH)
# encoder, decoder, discriminator = loadModel(PATH, epoch = 300)
