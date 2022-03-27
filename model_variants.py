import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F


class PATBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(PATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
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

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2),
                       nn.ReLU(True)]
        else:
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

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = F.sigmoid(x2_out)

        x1_out = x1_out * att
        out = x1 + x1_out

        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1_out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class PATNModel_attention_mask(nn.Module):
    def __init__(self, input_nc, output_nc, num_class, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0 and type(input_nc) == list)
        super(PATNModel_attention_mask, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        use_bias = True
        model_stream1_down = [nn.ReflectionPad2d(3), nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0, bias=use_bias),
                              norm_layer(ngf), nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3), nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0, bias=use_bias),
                              norm_layer(ngf), nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2), nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i]))

        mult = 2 ** (n_downsampling - 0)
        model_stream1_up_1 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                 padding=1, output_padding=1, bias=use_bias)]
        self.model_stream1_up_1_bn = ConditionalBatchNorm2d(int(ngf * mult / 2), num_class)

        mult = 2 ** (n_downsampling - 1)
        model_stream1_up_2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                 padding=1, output_padding=1, bias=use_bias)]
        self.model_stream1_up_2_bn = ConditionalBatchNorm2d(int(ngf * mult / 2), num_class)

        model_stream1_up = []
        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc+1, kernel_size=7, padding=0)]
        self.output_nc = output_nc

        self.model_stream1_down = nn.Sequential(*model_stream1_down)
        self.model_stream2_down = nn.Sequential(*model_stream2_down)
        self.att = attBlock
        self.model_stream1_up_1 = nn.Sequential(*model_stream1_up_1)
        self.model_stream1_up_2 = nn.Sequential(*model_stream1_up_2)
        self.model_stream1_up = nn.Sequential(*model_stream1_up)

    def forward(self, input, class_info):
        x1, x2 = input
        source_image = x1
        x1 = self.model_stream1_down(x1)
        x2 = self.model_stream2_down(x2)

        for model in self.att:
            x1, x2, _ = model(x1, x2)

        x1 = self.model_stream1_up_1(x1)
        x1 = self.model_stream1_up_1_bn.forward(x1, class_info)
        x1 = nn.ReLU(True)(x1)
        x1 = self.model_stream1_up_2(x1)
        x1 = self.model_stream1_up_2_bn.forward(x1, class_info)
        x1 = nn.ReLU(True)(x1)
        x1 = self.model_stream1_up(x1)

        output_image = x1[:, 0:self.output_nc, :, :]
        output_image = nn.Tanh()(output_image)
        mask = x1[:, self.output_nc:self.output_nc+1, :, :]
        mask = nn.Sigmoid()(mask)
        mask = mask.expand_as(output_image)
        final_output = output_image * mask + source_image * (1-mask)

        return final_output


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

