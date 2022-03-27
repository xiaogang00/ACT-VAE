from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.autograd import grad
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


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


def util_cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


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


class ResnetDiscriminator(nn.Module):
    def __init__(self, input_nc, num_class, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', use_sigmoid=False, n_downsampling=2):
        assert (n_blocks >= 0)
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]

        self.n_downsampling = n_downsampling
        if n_downsampling <= 2:
            mult = 2 ** 0
            model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

            mult = 2 ** 1
            model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 1
            model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 2
            model4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        if n_downsampling <= 2:
            mult = 4
        else:
            mult = 8

        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]
        if n_downsampling <= 2:
            self.model1 = nn.Sequential(*model1)
            self.model2 = nn.Sequential(*model2)
            self.model3 = nn.Sequential(*model3)
            self.model_res = nn.Sequential(*model_res)
        else:
            self.model1 = nn.Sequential(*model1)
            self.model2 = nn.Sequential(*model2)
            self.model3 = nn.Sequential(*model3)
            self.model4 = nn.Sequential(*model4)
            self.model_res = nn.Sequential(*model_res)
        self.gan_type = 'lsgan'

    def forward(self, input, class_info):
        output_feature_list = []
        out_feature = self.model1(input)
        if self.n_downsampling <= 2:
            out_feature = self.model2(out_feature)
            output_feature_list.append(out_feature)

            out_feature = self.model3(out_feature)
            output_feature_list.append(out_feature)
        else:
            out_feature = self.model2(out_feature)
            output_feature_list.append(out_feature)

            out_feature = self.model3(out_feature)
            output_feature_list.append(out_feature)

            out_feature = self.model4(out_feature)
            output_feature_list.append(out_feature)

        out_real_fake = self.model_res(out_feature)
        return out_real_fake, output_feature_list

    def gradient_penalty(self, x, y, class_info):
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = util_cuda(torch.rand(shape))
        z = x + alpha * (y - x)

        z = util_cuda(Variable(z, requires_grad=True))
        o = self.forward(z, class_info)[0]
        g = grad(o, z, grad_outputs=util_cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
        return gp

    def calc_dis_loss(self, input_fake, input_fake_class, input_real, input_real_class):
        outs0, _ = self.forward(input_fake, input_fake_class)
        outs1, _ = self.forward(input_real, input_real_class)
        loss = 0

        if self.gan_type == 'lsgan':
            loss += torch.mean((outs0 - 0) ** 2) + torch.mean((outs1 - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(outs0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(outs1.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(outs0), all0) +
                               F.binary_cross_entropy(F.sigmoid(outs1), all1))
        elif self.gan_type == 'wgan':
            wd = torch.mean(outs1) - torch.mean(outs0)
            gp = self.gradient_penalty(input_real.data, input_fake.data, input_real_class)
            loss += -wd + gp * 0.00001
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, input_fake_class):
        outs0, _ = self.forward(input_fake, input_fake_class)
        loss = 0
        if self.gan_type == 'lsgan':
            loss += torch.mean((outs0 - 1) ** 2)  # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(outs0.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(outs0), all1))
        elif self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


from model_variants import PATNModel_attention_mask
class Action_Generator(nn.Module):
    def __init__(self, encoder_params, vae_params, dec_im_params, dis_params):
        super(Action_Generator, self).__init__()

        self.n_label = dis_params['n_label']
        self.num_keypoint = 13
        self.colors = get_n_colors(self.num_keypoint)

        z_dim = 512
        self.z_dim = z_dim
        ###############################################################################################################
        self.image_enc_vae = RNN_ENCODER_VAE2(ninput=self.num_keypoint * 2 + self.n_label + z_dim,
                                              nhidden=z_dim*2, nlayers=1, input_dim=256)
        print('z_dim', z_dim)
        print('n_label', self.n_label)
        
        ###############################################################################################################
        self.image_enc_content = RNN_ENCODER_Decoder2(ninput=self.num_keypoint * 2 + self.n_label + z_dim,
                                                      nhidden=self.num_keypoint * 2,
                                                      input_dim=256, output_dim=256, nlayers=1)

        ###############################################################################################################
        input_nc = [3, self.num_keypoint * 2]
        self.generator = PATNModel_attention_mask(input_nc, 3, num_class=self.n_label, ngf=64, norm_layer=nn.BatchNorm2d,n_blocks=9)
        ###############################################################################################################
        input_dim_dis = dis_params['input_dim']
        n_label = dis_params['n_label']
        ndf = 64
        self.d_n_downsampling = 3
        self.classifier = ResnetDiscriminator(input_dim_dis + self.num_keypoint, n_label, ndf,
                                              norm_layer=nn.BatchNorm2d,
                                              use_dropout=False, n_blocks=3, gpu_ids=[],
                                              padding_type='reflect', use_sigmoid=False,
                                              n_downsampling=self.d_n_downsampling)

        self.criterionVGG = VGGLoss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_weight(self, input, target, weight_map):
        weight = weight_map[:, 0:1, :, :]
        distance = torch.abs(input - target)
        weight = weight.expand_as(distance)
        distance = (1 + weight * 10) * distance
        loss = torch.mean(distance)
        return loss

    def recon_criterion_L2(self, input, target):
        return torch.mean((input-target) ** 2)


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
        with torch.no_grad():
            batch_size = input1.shape[0]
            z_t_vae = self.image_enc_vae.get_z_random(batch_size, self.image_enc_vae.z_dim)
            z_t_vae = z_t_vae.unsqueeze(0)
            z_t_vae = z_t_vae.view(1, batch_size, self.image_enc_vae.z_dim)
            hidden_vae = self.image_enc_vae.init_hidden(batch_size)
            output_list = []
            mu_list = []
            std_list = []

            z_t_vae, hidden_vae, mu, logvar = self.image_enc_vae.forward_test(input1, hidden_vae, z_t_vae)
            hidden_d = self.image_enc_content.init_hidden(batch_size)
            output_pose, hidden_d = self.image_enc_content.forward_test(input1, z_t_vae, hidden_d)
            output_list.append(output_pose)
            mu_list.append(mu)
            std_list.append(logvar)

            output_pose_input = torch.cat([output_pose[:, 0, :], input_class], dim=1).unsqueeze(1)
            z_t_vae, hidden_vae, mu, logvar = self.image_enc_vae.forward_test(output_pose_input, hidden_vae, z_t_vae)
            output_pose, hidden_d = self.image_enc_content.forward_test(output_pose_input, z_t_vae, hidden_d)
            output_list.append(output_pose)
            mu_list.append(mu)
            std_list.append(logvar)

            output_pose_input = torch.cat([output_pose[:, 0, :], input_class], dim=1).unsqueeze(1)
            z_t_vae, hidden_vae, mu, logvar = self.image_enc_vae.forward_test(output_pose_input, hidden_vae, z_t_vae)
            output_pose, hidden_d = self.image_enc_content.forward_test(output_pose_input, z_t_vae, hidden_d)
            output_list.append(output_pose)
            mu_list.append(mu)
            std_list.append(logvar)

            output_pose_input = torch.cat([output_pose[:, 0, :], input_class], dim=1).unsqueeze(1)
            z_t_vae, hidden_vae, mu, logvar = self.image_enc_vae.forward_test(output_pose_input, hidden_vae, z_t_vae)
            output_pose, hidden_d = self.image_enc_content.forward_test(output_pose_input, z_t_vae, hidden_d)
            output_list.append(output_pose)
            mu_list.append(mu)
            std_list.append(logvar)

            output_list = torch.cat(output_list, dim=1)
            mu_final = torch.cat(mu_list, dim=0)
            std_final = torch.cat(std_list, dim=0)
            output_list = output_list.detach()
        return output_list, mu_final, std_final

    def forward(self, source_image, target_image1, target_image2, target_image3, target_image4, target_image5,
                pose_source,
                pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_label, mode):
        if mode == 'gen_update_image':
            h = source_image.shape[2]
            w = source_image.shape[3]
            batch_size = source_image.shape[0]

            index = random.randint(0, 4)
            target_list = [target_image1, target_image2, target_image3, target_image4, target_image5]
            target_pose_list = [pose_target1, pose_target2, pose_target3, pose_target4, pose_target5]
            target_mask_list = [mask_image1, mask_image2, mask_image3, mask_image4, mask_image5]
            pose_target = target_pose_list[index]
            target_image = target_list[index]
            mask_image = target_mask_list[index]

            output_pose = self.get_gaussian_maps(pose_target, h, w)
            source_pose = self.get_gaussian_maps(pose_source, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image = self.generator.forward(G_input, action_label.clone().view(batch_size, ).long())
            
            loss_recon_target = self.recon_criterion_weight(output_image, target_image, mask_image) * 30
            loss_vgg_target = self.criterionVGG.forward(output_image, target_image) * 5

            classifier_input = torch.cat([output_image, output_pose], dim=1)
            loss_r = self.classifier.calc_gen_loss(classifier_input, action_label)
            loss_gen_adv_pair = loss_r

            classifier_input_real = torch.cat([target_image, output_pose], dim=1)
            _, feature_fake = self.classifier.forward(classifier_input, action_label)
            _, feature_real = self.classifier.forward(classifier_input_real, action_label)

            loss_feature = self.recon_criterion(feature_fake[0], feature_real[0])
            if self.d_n_downsampling == 3:
                loss_feature += self.recon_criterion(feature_fake[1], feature_real[1])
                loss_feature += self.recon_criterion(feature_fake[2], feature_real[2])
            else:
                loss_feature += self.recon_criterion(feature_fake[1], feature_real[1])

            loss_gen_adv_pair = loss_gen_adv_pair * 5
            loss_gen_total = loss_recon_target + loss_vgg_target + loss_gen_adv_pair + loss_feature
            loss_gen_total.backward()
            return loss_recon_target, loss_vgg_target, loss_gen_adv_pair, loss_feature

        if mode == 'gen_update_image1':
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            #####################################################################
            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            output_pose1 = output_pose_all[:, 0, :]
            output_pose1 = output_pose1.view(batch_size, self.num_keypoint, 2)
            output_pose1 = self.get_gaussian_maps(output_pose1, h, w)
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose1), 1)]
            output_image = self.generator.forward(G_input, action_label.clone().view(batch_size, ).long())
            classifier_input = torch.cat([output_image, output_pose1], dim=1)
            loss_r = self.classifier.calc_gen_loss(classifier_input, action_label)
            loss_gen_adv_pair = loss_r

            loss_gen_adv_pair = loss_gen_adv_pair * 5
            loss_gen_total = loss_gen_adv_pair
            loss_gen_total.backward()
            return loss_gen_adv_pair

        if mode == 'gen_update_image2':
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            #####################################################################
            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)

            output_pose2 = output_pose_all[:, 1, :]
            output_pose2 = output_pose2.view(batch_size, self.num_keypoint, 2)
            output_pose2 = self.get_gaussian_maps(output_pose2, h, w)
            G_input2 = [source_image, torch.cat((source_pose, output_pose2), 1)]
            output_image2 = self.generator.forward(G_input2, action_label.clone().view(batch_size, ).long())
            classifier_input2 = torch.cat([output_image2, output_pose2], dim=1)
            loss_r2 = self.classifier.calc_gen_loss(classifier_input2, action_label)
            loss_gen_adv_pair = loss_r2

            loss_gen_adv_pair = loss_gen_adv_pair * 5
            loss_gen_total = loss_gen_adv_pair
            loss_gen_total.backward()
            return loss_gen_adv_pair

        if mode == 'gen_update_image3':
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            #####################################################################
            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)

            output_pose3 = output_pose_all[:, 2, :]
            output_pose3 = output_pose3.view(batch_size, self.num_keypoint, 2)
            output_pose3 = self.get_gaussian_maps(output_pose3, h, w)
            G_input3 = [source_image, torch.cat((source_pose, output_pose3), 1)]
            output_image3 = self.generator.forward(G_input3, action_label.clone().view(batch_size, ).long())
            classifier_input3 = torch.cat([output_image3, output_pose3], dim=1)
            loss_r3 = self.classifier.calc_gen_loss(classifier_input3, action_label)
            loss_gen_adv_pair = loss_r3

            loss_gen_adv_pair = loss_gen_adv_pair * 5
            loss_gen_total = loss_gen_adv_pair
            loss_gen_total.backward()
            return loss_gen_adv_pair

        if mode == 'gen_update_image4':
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            #####################################################################
            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)

            output_pose4 = output_pose_all[:, 3, :]
            output_pose4 = output_pose4.view(batch_size, self.num_keypoint, 2)
            output_pose4 = self.get_gaussian_maps(output_pose4, h, w)
            G_input4 = [source_image, torch.cat((source_pose, output_pose4), 1)]
            output_image4 = self.generator.forward(G_input4, action_label.clone().view(batch_size, ).long())
            classifier_input4 = torch.cat([output_image4, output_pose4], dim=1)
            loss_r4 = self.classifier.calc_gen_loss(classifier_input4, action_label)
            loss_gen_adv_pair = loss_r4

            loss_gen_adv_pair = loss_gen_adv_pair * 5
            loss_gen_total = loss_gen_adv_pair
            loss_gen_total.backward()
            return loss_gen_adv_pair

        if mode == 'class_update':
            ######################################################################################################
            batch_size = source_image.shape[0]
            h = source_image.shape[2]
            w = source_image.shape[3]

            index = random.randint(0, 4)
            target_list = [target_image1, target_image2, target_image3, target_image4, target_image5]
            target_pose_list = [pose_target1, pose_target2, pose_target3, pose_target4, pose_target5]
            pose_target = target_pose_list[index]
            target_image = target_list[index]

            output_pose = self.get_gaussian_maps(pose_target, h, w)
            source_pose = self.get_gaussian_maps(pose_source, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image = self.generator.forward(G_input, action_label.clone().view(batch_size, ).long())

            classifier_input_real = torch.cat([target_image, output_pose], dim=1)
            classifier_input_fake = torch.cat([output_image, output_pose], dim=1)
            loss_r = self.classifier.calc_dis_loss(classifier_input_fake, action_label,
                                                   classifier_input_real, action_label)
            loss_dis_pair = loss_r

            loss_dis_pair = loss_dis_pair * 5
            loss_all = loss_dis_pair
            loss_all.backward()
            return loss_dis_pair

        if mode == 'class_update1':
            ######################################################################################################
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            output_pose1 = output_pose_all[:, 0, :]
            output_pose1 = output_pose1.view(batch_size, self.num_keypoint, 2)
            output_pose1 = self.get_gaussian_maps(output_pose1, h, w)
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose1), 1)]
            output_image1 = self.generator.forward(G_input, action_label.clone().view(batch_size, ).long())

            pose_target_1 = pose_target1.view(batch_size, self.num_keypoint, 2)
            output_pose_target_1 = self.get_gaussian_maps(pose_target_1, h, w)
            classifier_input_real1 = torch.cat([target_image1, output_pose_target_1], dim=1)
            classifier_input_fake1 = torch.cat([output_image1, output_pose1], dim=1)
            loss_r = self.classifier.calc_dis_loss(classifier_input_fake1, action_label,
                                                   classifier_input_real1, action_label)
            loss_dis_pair = loss_r

            loss_dis_pair = loss_dis_pair * 5
            loss_all = loss_dis_pair
            loss_all.backward()
            return loss_dis_pair

        if mode == 'class_update2':
            ######################################################################################################
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)
            #################################################################
            output_pose2 = output_pose_all[:, 1, :]
            output_pose2 = output_pose2.view(batch_size, self.num_keypoint, 2)
            output_pose2 = self.get_gaussian_maps(output_pose2, h, w)
            G_input2 = [source_image, torch.cat((source_pose, output_pose2), 1)]
            output_image2 = self.generator.forward(G_input2, action_label.clone().view(batch_size, ).long())

            pose_target_2 = pose_target2.view(batch_size, self.num_keypoint, 2)
            output_pose_target_2 = self.get_gaussian_maps(pose_target_2, h, w)
            classifier_input_real2 = torch.cat([target_image2, output_pose_target_2], dim=1)
            classifier_input_fake2 = torch.cat([output_image2, output_pose2], dim=1)
            loss_r2 = self.classifier.calc_dis_loss(classifier_input_fake2, action_label,
                                                    classifier_input_real2, action_label)
            loss_dis_pair = loss_r2

            loss_dis_pair = loss_dis_pair * 5
            loss_all = loss_dis_pair
            loss_all.backward()
            return loss_dis_pair

        if mode == 'class_update3':
            ######################################################################################################
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)
            output_pose3 = output_pose_all[:, 2, :]
            output_pose3 = output_pose3.view(batch_size, self.num_keypoint, 2)
            output_pose3 = self.get_gaussian_maps(output_pose3, h, w)
            G_input3 = [source_image, torch.cat((source_pose, output_pose3), 1)]
            output_image3 = self.generator.forward(G_input3, action_label.clone().view(batch_size, ).long())

            pose_target_3 = pose_target3.view(batch_size, self.num_keypoint, 2)
            output_pose_target_3 = self.get_gaussian_maps(pose_target_3, h, w)
            classifier_input_real3 = torch.cat([target_image3, output_pose_target_3], dim=1)
            classifier_input_fake3 = torch.cat([output_image3, output_pose3], dim=1)
            loss_r3 = self.classifier.calc_dis_loss(classifier_input_fake3, action_label,
                                                    classifier_input_real3, action_label)
            loss_dis_pair = loss_r3

            loss_dis_pair = loss_dis_pair * 5
            loss_all = loss_dis_pair
            loss_all.backward()
            return loss_dis_pair

        if mode == 'class_update4':
            ######################################################################################################
            batch_size = pose_source.shape[0]
            pose_1 = pose_source.view(batch_size, -1)
            input_class = self.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

            output_pose_all, _, _ = self.forward_generate(input1, input_class)

            h = source_image.shape[2]
            w = source_image.shape[3]
            source_pose = pose_source.view(batch_size, self.num_keypoint, 2)
            source_pose = self.get_gaussian_maps(source_pose, h, w)
            #################################################################
            output_pose4 = output_pose_all[:, 3, :]
            output_pose4 = output_pose4.view(batch_size, self.num_keypoint, 2)
            output_pose4 = self.get_gaussian_maps(output_pose4, h, w)
            G_input4 = [source_image, torch.cat((source_pose, output_pose4), 1)]
            output_image4 = self.generator.forward(G_input4, action_label.clone().view(batch_size, ).long())

            pose_target_4 = pose_target4.view(batch_size, self.num_keypoint, 2)
            output_pose_target_4 = self.get_gaussian_maps(pose_target_4, h, w)
            classifier_input_real4 = torch.cat([target_image4, output_pose_target_4], dim=1)
            classifier_input_fake4 = torch.cat([output_image4, output_pose4], dim=1)
            loss_r4 = self.classifier.calc_dis_loss(classifier_input_fake4, action_label,
                                                    classifier_input_real4, action_label)
            loss_dis_pair = loss_r4

            loss_dis_pair = loss_dis_pair * 5
            loss_all = loss_dis_pair
            loss_all.backward()
            return loss_dis_pair

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
