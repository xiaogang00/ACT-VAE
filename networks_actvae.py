from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import functools
try:
    from itertools import izip as zip
except ImportError:
    pass


from vgg import VGG_Activations2
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG_Activations2([1, 6, 11, 20, 29]).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def compute_loss(self, x_vgg, y_vgg):
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            x, y = x.view(-1, c, h, w), y.view(-1, c, h, w)

        y_vgg = self.vgg(y)
        x_vgg = self.vgg(x)
        loss = self.compute_loss(x_vgg, y_vgg)
        return loss



def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for i in range(n):
        colors.append(generate_new_color(colors, pastel_factor=0.9))
    return colors


class Action_Generator(nn.Module):
    def __init__(self, encoder_params, vae_params, dec_im_params, dis_params):
        super(Action_Generator, self).__init__()

        self.n_label = dis_params['n_label']
        self.num_keypoint = 13
        self.colors = get_n_colors(self.num_keypoint)

        z_dim = 512
        self.z_dim = z_dim
        self.image_enc_vae = RNN_ENCODER_VAE2(ninput=self.num_keypoint * 2 + self.n_label + z_dim,
                                              nhidden=z_dim*2, nlayers=1, input_dim=256)
        print('z_dim', z_dim)
        print('n_label', self.n_label)
        ###############################################################################################################
        self.image_enc_content = RNN_ENCODER_Decoder2(ninput=self.num_keypoint * 2 + self.n_label + z_dim,
                                                      nhidden=self.num_keypoint * 2,
                                                      input_dim=256, output_dim=256, nlayers=1)
        ###############################################################################################################

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def kl_loss(self, logvar, mu, lambda_kl=0.002):
        # 0.002 100
        # 0.0002 10
        loss = torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (-0.5 * lambda_kl)
        return loss

    def obtain_action_label_feature2(self, action_label, batch_size):
        assert action_label.size(0) == batch_size
        action_label_clone = action_label.view(batch_size, )
        action_label_feature = torch.zeros(batch_size, self.n_label).cuda().float()
        index = torch.LongTensor(range(action_label_feature.size(0))).cuda()
        action_label_feature[index, action_label_clone.long()] = 1.0
        return action_label_feature

    def get_gaussian_maps(self, mu, h, w, inv_std=14.3):
        batch_size = mu.shape[0]
        mu_x = mu[:, :, 0]
        mu_y = mu[:, :, 1]
        # mu_x:-1 to 1, mu_y:-1 to 1
        num_keypoint = self.num_keypoint
        mu_x = mu_x.view(batch_size, num_keypoint, 1, 1)
        mu_y = mu_y.view(batch_size, num_keypoint, 1, 1)
        mu_x = mu_x.repeat(1, 1, h, w)
        mu_y = mu_y.repeat(1, 1, h, w)

        y = np.linspace(-1.0, 1.0, h)
        x = np.linspace(-1.0, 1.0, w)
        y = np.repeat(np.reshape(y, [1, h, 1]), num_keypoint, axis=0)
        x = np.repeat(np.reshape(x, [1, 1, w]), num_keypoint, axis=0)
        y = np.repeat(y, w, axis=2)
        x = np.repeat(x, h, axis=1)
        y = np.repeat(np.expand_dims(y, axis=0), batch_size, axis=0)
        x = np.repeat(np.expand_dims(x, axis=0), batch_size, axis=0)
        y = torch.Tensor(y).float().cuda()
        x = torch.Tensor(x).float().cuda()

        g_y = (y - mu_y) ** 2
        g_x = (x - mu_x) ** 2
        dist = (g_y + g_x) * inv_std ** 2
        maps = torch.exp(-dist)
        return maps

    def visual_map(self, maps, num_keypoint=13):
        maps = maps.permute(1, 2, 0).clone().detach().cpu().numpy()
        hmaps = [np.expand_dims(maps[..., i], axis=2) * np.reshape(self.colors[i], [1, 1, 3])
                 for i in range(num_keypoint)]
        result = np.max(hmaps, axis=0)
        return result

    def forward_generate(self, input1, input_class):
        batch_size = input1.shape[0]
        z_t_vae0 = self.image_enc_vae.get_z_random(batch_size, self.image_enc_vae.z_dim)
        z_t_vae0 = z_t_vae0.unsqueeze(0)
        z_t_vae0 = z_t_vae0.view(1, batch_size, self.image_enc_vae.z_dim)
        hidden_vae0 = self.image_enc_vae.init_hidden(batch_size)
        output_list = []
        mu_list = []
        std_list = []

        z_t_vae1, hidden_vae1, mu1, logvar1 = self.image_enc_vae.forward_test(input1, hidden_vae0, z_t_vae0)
        hidden_d0 = self.image_enc_content.init_hidden(batch_size)
        output_pose1, hidden_d1 = self.image_enc_content.forward_test(input1, z_t_vae1, hidden_d0)
        output_list.append(output_pose1)
        mu_list.append(mu1)
        std_list.append(logvar1)

        output_pose_input1 = torch.cat([output_pose1[:, 0, :], input_class], dim=1).unsqueeze(1)
        z_t_vae2, hidden_vae2, mu2, logvar2 = self.image_enc_vae.forward_test(output_pose_input1, hidden_vae1, z_t_vae1)
        output_pose2, hidden_d2 = self.image_enc_content.forward_test(output_pose_input1, z_t_vae2, hidden_d1)
        output_list.append(output_pose2)
        mu_list.append(mu2)
        std_list.append(logvar2)

        output_pose_input2 = torch.cat([output_pose2[:, 0, :], input_class], dim=1).unsqueeze(1)
        z_t_vae3, hidden_vae3, mu3, logvar3 = self.image_enc_vae.forward_test(output_pose_input2, hidden_vae2, z_t_vae2)
        output_pose3, hidden_d3 = self.image_enc_content.forward_test(output_pose_input2, z_t_vae3, hidden_d2)
        output_list.append(output_pose3)
        mu_list.append(mu3)
        std_list.append(logvar3)

        output_pose_input3 = torch.cat([output_pose3[:, 0, :], input_class], dim=1).unsqueeze(1)
        z_t_vae4, hidden_vae4, mu4, logvar4 = self.image_enc_vae.forward_test(output_pose_input3, hidden_vae3, z_t_vae3)
        output_pose4, hidden_d4 = self.image_enc_content.forward_test(output_pose_input3, z_t_vae4, hidden_d3)
        output_list.append(output_pose4)
        mu_list.append(mu4)
        std_list.append(logvar4)

        output_list = torch.cat(output_list, dim=1)
        mu_final = torch.cat(mu_list, dim=0)
        std_final = torch.cat(std_list, dim=0)
        return output_list, mu_final, std_final

    def forward(self, source_image, target_image1, target_image2, target_image3, target_image4, target_image5,
                pose_source,
                pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_label, mode):
        if mode == 'gen_update_image1':
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            pose_2 = pose_target1.view(batch_size, -1)
            pose_3 = pose_target2.view(batch_size, -1)
            pose_4 = pose_target3.view(batch_size, -1)
            pose_5 = pose_target4.view(batch_size, -1)
            pose_6 = pose_target5.view(batch_size, -1)

            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            target_pose1 = torch.cat([pose_2.unsqueeze(1), pose_3.unsqueeze(1),
                                      pose_4.unsqueeze(1), pose_5.unsqueeze(1)], dim=1)
            target_pose2 = torch.cat([pose_2.unsqueeze(1), pose_3.unsqueeze(1),
                                      pose_4.unsqueeze(1), pose_6.unsqueeze(1)], dim=1)
            target_pose3 = torch.cat([pose_3.unsqueeze(1), pose_4.unsqueeze(1),
                                      pose_5.unsqueeze(1), pose_6.unsqueeze(1)], dim=1)
            
            #####################################################################
            output_pose, mu_final, logvar_final = self.forward_generate(input1, input_class)
            loss_KL = self.kl_loss(logvar_final, mu_final)
            loss_recon_pose = self.recon_criterion(output_pose, target_pose1) * 200
            
            #####################################################################
            output_pose2, _, _ = self.forward_generate(input1, input_class)
            loss_recon_pose += self.recon_criterion(output_pose2, target_pose2) * 200

            #####################################################################
            output_pose3, _, _ = self.forward_generate(input1, input_class)
            loss_recon_pose += self.recon_criterion(output_pose3, target_pose3) * 200

            loss_gen_total = loss_recon_pose + loss_KL
            loss_gen_total.backward()
            return loss_KL, loss_recon_pose
        else:
            assert 0, 'Not support operation'


class RNN_ENCODER_VAE2(nn.Module):
    def __init__(self, ninput=300, nhidden=128, nlayers=1, input_dim=256):
        super(RNN_ENCODER_VAE2, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden

        self.nlayers = nlayers
        self.num_directions = 1
        self.z_dim = self.nhidden // 2

        self.drop_prob = 0.2
        self.define_module()

    def define_module(self):
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                           self.nlayers, batch_first=False,
                           dropout=self.drop_prob,
                           bidirectional=False)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * self.num_directions,
                                    bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions,
                                    bsz, self.nhidden).zero_()))

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        z = z.cuda()
        return z

    def forward_test(self, emb, hidden, z_t):
        # input: torch.LongTensor of size batch x n_steps
        # emb: batch * seq * dim
        # z_t: seq * batch * dim

        emb = emb.permute(1, 0, 2)
        # seq * batch * dim
        input_variable = emb
        input_variable_fusion = torch.cat([input_variable, z_t], dim=2)
        output, hidden = self.rnn(input_variable_fusion, hidden)

        mu, logvar = output[0, :, 0:self.z_dim], output[0, :, self.z_dim:2 * self.z_dim]
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z_t = eps.mul(std).add_(mu)
        z_t = z_t.unsqueeze(0)
        return z_t, hidden, mu, logvar


class RNN_ENCODER_Decoder2(nn.Module):
    def __init__(self, ninput=300, nhidden=128, input_dim=256, output_dim=256, nlayers=1):
        super(RNN_ENCODER_Decoder2, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden

        self.nlayers = nlayers
        self.num_directions = 1

        self.drop_prob = 0.2

        self.define_module()

    def define_module(self):
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                           self.nlayers, batch_first=False,
                           dropout=self.drop_prob,
                           bidirectional=False)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * self.num_directions,
                                    bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions,
                                    bsz, self.nhidden).zero_()))

    def forward_test(self, emb_this, z_this, hidden):
        # input: torch.LongTensor of size batch x n_steps
        # emb: batch * seq * dim
        emb_this = emb_this.permute(1, 0, 2)
        # seq * batch * dim

        emb_final = torch.cat([emb_this, z_this], dim=2)
        output, hidden = self.rnn(emb_final, hidden)
        # seq_len, batch, dim
        output = output.permute(1, 0, 2)
        # batch * seq_len * dim
        return output, hidden


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'in_sn':
            self.norm = nn.InstanceNorm2d(norm_dim)
            norm = 'sn'
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.activation_first = activation_first
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.squeeze(1)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        # return relu5_3
        return [relu1_2, relu2_2, relu3_3, relu4_3, relu5_3]

    def forward_single(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        # return relu5_3
        return relu2_2
