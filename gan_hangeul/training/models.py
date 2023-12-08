import torch
import torch.nn as nn
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        cs = 64
        self.e1 = nn.Sequential(nn.Conv2d(1, cs, 5, 2, 2))

        self.e2 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs, cs*2, 5, 2, 2))

        self.e3 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*2, cs*4, 5, 2, 2),
                                nn.BatchNorm2d(cs*4))

        self.e4 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*4, cs*8, 5, 2, 2),
                                nn.BatchNorm2d(cs*8))

        self.e5 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*8, cs*8, 5, 2, 2),
                                nn.BatchNorm2d(cs*8))

        self.e6 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*8, cs*8, 5, 2, 2),
                                nn.BatchNorm2d(cs*8))

        self.e7 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*8, cs*8, 5, 2, 2))

        self.e1.apply(init_weights)
        self.e2.apply(init_weights)
        self.e3.apply(init_weights)
        self.e4.apply(init_weights)
        self.e5.apply(init_weights)
        self.e6.apply(init_weights)
        self.e7.apply(init_weights)

    def forward(self, x):
        d = dict()
        x = self.e1(x)
        d['e1'] = x
        x = self.e2(x)
        d['e2'] = x
        x = self.e3(x)
        d['e3'] = x
        x = self.e4(x)
        d['e4'] = x
        x = self.e5(x)
        d['e5'] = x
        x = self.e6(x)
        d['e6'] = x
        x = self.e7(x)
        d['e7'] = x
        return x, d


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        cs = 64
        # 128 = style vector
        self.d1 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*8+128, cs*8, 3, 1, 1), # e1이랑 합치기위해 유지
                                nn.BatchNorm2d(cs*8),
                                nn.Dropout(0.5))
        self.d2 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*16, cs*8, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs*8),
                                nn.Dropout(0.5))
        self.d3 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*16, cs*8, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs*8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*16, cs*8, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs*8))
        self.d5 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*16, cs*4, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs*4))
        self.d6 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*8, cs*2, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs*2))
        self.d7 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*4, cs, 5, 2, 2, 1),
                                nn.BatchNorm2d(cs))
        self.d8 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(cs*2, 1, 5, 2, 2, 1),
                                nn.Tanh())

        self.d1.apply(init_weights)
        self.d2.apply(init_weights)
        self.d3.apply(init_weights)
        self.d4.apply(init_weights)
        self.d5.apply(init_weights)
        self.d6.apply(init_weights)
        self.d7.apply(init_weights)
        self.d8.apply(init_weights)

    def forward(self, x, e):
        x = self.d1(x)
        x = torch.cat((x, e['e7']), dim=1)
        x = self.d2(x)
        x = torch.cat((x, e['e6']), dim=1)
        x = self.d3(x)
        x = torch.cat((x, e['e5']), dim=1)
        x = self.d4(x)
        x = torch.cat((x, e['e4']), dim=1)
        x = self.d5(x)
        x = torch.cat((x, e['e3']), dim=1)
        x = self.d6(x)
        x = torch.cat((x, e['e2']), dim=1)
        x = self.d7(x)
        x = torch.cat((x, e['e1']), dim=1)
        x = self.d8(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, category_num):
        super(Discriminator, self).__init__()
        cs = 64
        self.category_num = category_num
        self.d1 = nn.Sequential(nn.Conv2d(1, cs, 4, 2, 1))
        self.d2 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs, cs*2, 4, 2, 1),
                                nn.BatchNorm2d(cs*2))
        self.d3 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*2, cs*4, 4, 2, 1),
                                nn.BatchNorm2d(cs*4))
        self.d4 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(cs*4, cs*8, 4, 2, 1),
                                nn.BatchNorm2d(cs*8))
        self.fc_tf = nn.Sequential(nn.Flatten(),
                                   nn.Linear(cs*8*8*8, 1))
        self.fc_cg = nn.Sequential(nn.Flatten(),
                                   nn.Linear(cs*8*8*8, category_num))

        self.d1.apply(init_weights)
        self.d2.apply(init_weights)
        self.d3.apply(init_weights)
        self.d4.apply(init_weights)
        self.fc_tf.apply(init_weights)
        self.fc_cg.apply(init_weights)

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        tf = self.fc_tf(x)
        cg = self.fc_cg(x)
        return tf, cg
