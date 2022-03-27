from networks_actvae_p2inetwork import Action_Generator
from utils_file import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import random
import time


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        lr = hyperparameters['lr']

        encoder_params = hyperparameters['image_encode']
        vae_params = hyperparameters['image_vae']
        dec_im_params = hyperparameters['dec_im']
        dis_params = hyperparameters['dis']
        self.model = Action_Generator(encoder_params, vae_params, dec_im_params, dis_params)
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        #####################################################
        generator_params = list(self.model.generator.parameters())
        self.generator_opt = torch.optim.Adam([p for p in generator_params if p.requires_grad],
                                              lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.generator_scheduler = get_scheduler(self.generator_opt, hyperparameters)
        #####################################################

        #####################################################
        classifier_params = list(self.model.classifier.parameters())
        self.classifier_opt = torch.optim.Adam([p for p in classifier_params if p.requires_grad],
                                               lr=lr, betas=(beta1, beta2),
                                               weight_decay=hyperparameters['weight_decay'])
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters)
        #####################################################
        #####################################################
        # Network weight initialization
        self.model.classifier.apply(weights_init('gaussian'))
        self.model.generator.apply(weights_init('gaussian'))

    def gen_update(self, source_image, target_image1, target_image2, target_image3, target_image4, target_image5, 
                   mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_label, 
                   pose_source, pose_target1, pose_target2, pose_target3, pose_target4, pose_target5):
        self.classifier_opt.zero_grad()
        self.generator_opt.zero_grad()

        loss_recon_target, loss_vgg_target, \
        loss_gen_adv_pair, loss_feature = self.model(source_image,
                                                     target_image1, target_image2, target_image3, target_image4,
                                                     target_image5,
                                                     pose_source,
                                                     pose_target1, pose_target2, pose_target3, pose_target4,
                                                     pose_target5,
                                                     mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                                     action_label, 'gen_update_image')
        self.loss_recon_target = torch.mean(loss_recon_target)
        self.loss_vgg_target = torch.mean(loss_vgg_target)
        self.loss_gen_adv_pair = torch.mean(loss_gen_adv_pair)
        self.loss_feature = torch.mean(loss_feature)
        self.generator_opt.step()

        #####################################################
        if self.random_index == 0:
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_gen_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4,
                                            target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4,
                                            pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'gen_update_image1')
            self.loss_gen_adv_pair2 = torch.mean(loss_gen_adv_pair2)
            self.generator_opt.step()

        elif self.random_index == 1:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_gen_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4,
                                            target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4,
                                            pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'gen_update_image2')
            self.loss_gen_adv_pair2 = torch.mean(loss_gen_adv_pair2)
            self.generator_opt.step()

        elif self.random_index == 2:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_gen_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4,
                                            target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4,
                                            pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'gen_update_image3')
            self.loss_gen_adv_pair2 = torch.mean(loss_gen_adv_pair2)
            self.generator_opt.step()

        elif self.random_index == 3:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_gen_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4,
                                            target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4,
                                            pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'gen_update_image4')
            self.loss_gen_adv_pair2 = torch.mean(loss_gen_adv_pair2)
            self.generator_opt.step()

    def dis_update(self, source_image, target_image1, target_image2, target_image3, target_image4, target_image5, 
                   mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_label, 
                   pose_source, pose_target1, pose_target2, pose_target3, pose_target4, pose_target5):
        self.classifier_opt.zero_grad()
        self.generator_opt.zero_grad()

        loss_dis_adv_pair = self.model(source_image,
                                       target_image1, target_image2, target_image3, target_image4, target_image5,
                                       pose_source,
                                       pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                                       mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                       action_label, 'class_update')
        self.loss_dis_adv_pair = torch.mean(loss_dis_adv_pair)
        self.classifier_opt.step()

        self.random_index = random.randint(0, 3)
        if self.random_index == 0:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_dis_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4, target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'class_update1')
            self.loss_dis_adv_pair2 = torch.mean(loss_dis_adv_pair2)
            self.classifier_opt.step()

        elif self.random_index == 1:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_dis_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4, target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'class_update2')
            self.loss_dis_adv_pair2 = torch.mean(loss_dis_adv_pair2)
            self.classifier_opt.step()

        elif self.random_index == 2:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_dis_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4, target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'class_update3')
            self.loss_dis_adv_pair2 = torch.mean(loss_dis_adv_pair2)
            self.classifier_opt.step()

        elif self.random_index == 3:
            #####################################################
            self.classifier_opt.zero_grad()
            self.generator_opt.zero_grad()

            loss_dis_adv_pair2 = self.model(source_image,
                                            target_image1, target_image2, target_image3, target_image4, target_image5,
                                            pose_source,
                                            pose_target1, pose_target2, pose_target3, pose_target4, pose_target5,
                                            mask_image1, mask_image2, mask_image3, mask_image4, mask_image5,
                                            action_label, 'class_update4')
            self.loss_dis_adv_pair2 = torch.mean(loss_dis_adv_pair2)
            self.classifier_opt.step()

    def test_forward(self, source_image, target_image1, target_image2, target_image3, target_image4, target_image5,
                     mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_label,
                     pose_source, pose_target1, pose_target2, pose_target3, pose_target4, pose_target5):

        self.eval()
        this_model = self.model.module
        this_model.generator.eval()

        x_pose_random = []
        x_img_random = []
        x_pose_real = []
        x_img_real = []

        pose_source_original = pose_source.clone()
        batch_size = source_image.shape[0]
        h = source_image.shape[2]
        w = source_image.shape[3]

        input_class = this_model.obtain_action_label_feature2(action_label, batch_size)
        pose_1 = pose_source.view(batch_size, -1)
        input1 = torch.cat([pose_1, input_class], dim=1).unsqueeze(1)

        pose_source_original_generate = this_model.get_gaussian_maps(pose_source_original, h, w)
        output_pose_generate = this_model.visual_map(pose_source_original_generate[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        x_pose_random.append(output_pose_generate)
        x_img_random.append(source_image)
        x_pose_real.append(output_pose_generate)
        x_img_real.append(source_image)

        pose_target1 = this_model.get_gaussian_maps(pose_target1, h, w)
        output_pose_generate = this_model.visual_map(pose_target1[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        x_pose_real.append(output_pose_generate)
        x_img_real.append(target_image1)

        pose_target2 = this_model.get_gaussian_maps(pose_target2, h, w)
        output_pose_generate = this_model.visual_map(pose_target2[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        x_pose_real.append(output_pose_generate)
        x_img_real.append(target_image2)

        pose_target3 = this_model.get_gaussian_maps(pose_target3, h, w)
        output_pose_generate = this_model.visual_map(pose_target3[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        x_pose_real.append(output_pose_generate)
        x_img_real.append(target_image3)

        pose_target4 = this_model.get_gaussian_maps(pose_target4, h, w)
        output_pose_generate = this_model.visual_map(pose_target4[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        x_pose_real.append(output_pose_generate)
        x_img_real.append(target_image4)

        ###################################################################################
        output_pose_all, _, _ = this_model.forward_generate(input1, input_class)

        output_pose = output_pose_all[:, 0, :].view(batch_size, this_model.num_keypoint, 2)
        output_pose = this_model.get_gaussian_maps(output_pose, h, w)
        output_pose_generate = this_model.visual_map(output_pose[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
        G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
        output_image_generate1_1 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
        x_pose_random.append(output_pose_generate)
        x_img_random.append(output_image_generate1_1)

        output_pose = output_pose_all[:, 1, :].view(batch_size, this_model.num_keypoint, 2)
        output_pose = this_model.get_gaussian_maps(output_pose, h, w)
        output_pose_generate = this_model.visual_map(output_pose[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
        G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
        output_image_generate1_2 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
        x_pose_random.append(output_pose_generate)
        x_img_random.append(output_image_generate1_2)

        output_pose = output_pose_all[:, 2, :].view(batch_size, this_model.num_keypoint, 2)
        output_pose = this_model.get_gaussian_maps(output_pose, h, w)
        output_pose_generate = this_model.visual_map(output_pose[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
        G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
        output_image_generate1_3 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
        x_pose_random.append(output_pose_generate)
        x_img_random.append(output_image_generate1_3)

        output_pose = output_pose_all[:, 3, :].view(batch_size, this_model.num_keypoint, 2)
        output_pose = this_model.get_gaussian_maps(output_pose, h, w)
        output_pose_generate = this_model.visual_map(output_pose[0])
        output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
        G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
        output_image_generate1_4 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
        x_pose_random.append(output_pose_generate)
        x_img_random.append(output_image_generate1_4)

        x_pose_random = torch.cat(x_pose_random)
        x_img_random = torch.cat(x_img_random)

        x_pose_real = torch.cat(x_pose_real)
        x_img_real = torch.cat(x_img_real)
        return x_pose_random, x_img_random, x_pose_real, x_img_real

    def sample(self, source_image_all,
               target_image_all1, target_image_all2, target_image_all3, target_image_all4, target_image_all5,
               mask_image_all1, mask_image_all2, mask_image_all3, mask_image_all4, mask_image_all5,
               action_label_all, source_pose_all,
               target_pose_all1, target_pose_all2, target_pose_all3, target_pose_all4, target_pose_all5):
        self.eval()
        this_model = self.model.module
        x_img_image = []
        x_img_random1_1 = []
        x_img_random1_2 = []
        x_img_random1_3 = []
        x_img_random1_4 = []
        x_img_random2_1 = []
        x_img_random2_2 = []
        x_img_random2_3 = []
        x_img_random2_4 = []
        x_pose_random1_1 = []
        x_pose_random1_2 = []
        x_pose_random1_3 = []
        x_pose_random1_4 = []
        x_pose_random2_1 = []
        x_pose_random2_2 = []
        x_pose_random2_3 = []
        x_pose_random2_4 = []

        x_distance1 = []
        x_distance2 = []
        x_distance3 = []
        x_distance4 = []

        x_save = []

        x_target1 = []
        x_target2 = []
        x_target3 = []
        x_target4 = []
        x_target5 = []

        for i in range(len(source_image_all)):
            source_image = source_image_all[i:i+1]
            target_image1 = target_image_all1[i:i+1]
            target_image2 = target_image_all2[i:i+1]
            target_image3 = target_image_all3[i:i+1]
            target_image4 = target_image_all4[i:i+1]
            target_image5 = target_image_all5[i:i+1]
            mask_image1 = mask_image_all1[i:i + 1]
            mask_image2 = mask_image_all2[i:i + 1]
            mask_image3 = mask_image_all3[i:i + 1]
            mask_image4 = mask_image_all4[i:i + 1]
            mask_image5 = mask_image_all5[i:i + 1]
            action_label = action_label_all[i:i+1]
            pose_source = source_pose_all[i:i + 1]
            pose_target1 = target_pose_all1[i:i + 1]
            pose_target2 = target_pose_all2[i:i + 1]
            pose_target3 = target_pose_all3[i:i + 1]
            pose_target4 = target_pose_all4[i:i + 1]
            pose_target5 = target_pose_all5[i:i + 1]
            pose_source_original = pose_source.clone()
            pose_target_original1 = pose_target1.clone()
            pose_target_original2 = pose_target2.clone()
            pose_target_original3 = pose_target3.clone()
            pose_target_original4 = pose_target4.clone()
            pose_target_original5 = pose_target5.clone()

            batch_size = pose_source.shape[0]
            pose_source1 = pose_source.view(batch_size, -1)
            h = source_image.shape[2]
            w = source_image.shape[3]
            input_class = this_model.obtain_action_label_feature2(action_label, batch_size)
            input1 = torch.cat([pose_source1, input_class], dim=1).unsqueeze(1)

            #########################################################################################################
            output_pose_all, _, _ = this_model.forward_generate(input1, input_class)

            output_pose = output_pose_all[:, 0, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate1_1 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random1_1.append(output_pose_generate)
            x_img_random1_1.append(output_image_generate1_1)

            output_pose = output_pose_all[:, 1, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate1_2 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random1_2.append(output_pose_generate)
            x_img_random1_2.append(output_image_generate1_2)

            output_pose = output_pose_all[:, 2, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate1_3 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random1_3.append(output_pose_generate)
            x_img_random1_3.append(output_image_generate1_3)

            output_pose = output_pose_all[:, 3, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate1_4 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random1_4.append(output_pose_generate)
            x_img_random1_4.append(output_image_generate1_4)

            #########################################################################################################
            output_pose_all2, _, _ = this_model.forward_generate(input1, input_class)

            output_pose = output_pose_all2[:, 0, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate2_1 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random2_1.append(output_pose_generate)
            x_img_random2_1.append(output_image_generate2_1)

            output_pose = output_pose_all2[:, 1, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate2_2 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random2_2.append(output_pose_generate)
            x_img_random2_2.append(output_image_generate2_2)

            output_pose = output_pose_all2[:, 2, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate2_3 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random2_3.append(output_pose_generate)
            x_img_random2_3.append(output_image_generate2_3)

            output_pose = output_pose_all2[:, 3, :].view(batch_size, this_model.num_keypoint, 2)
            output_pose = this_model.get_gaussian_maps(output_pose, h, w)
            output_pose_generate = this_model.visual_map(output_pose[0])
            output_pose_generate = torch.Tensor(output_pose_generate).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            source_pose = this_model.get_gaussian_maps(pose_source_original, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate2_4 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())
            x_pose_random2_4.append(output_pose_generate)
            x_img_random2_4.append(output_image_generate2_4)

            #########################################################################################################
            h = source_image.shape[2]
            w = source_image.shape[3]
            output_pose = this_model.get_gaussian_maps(pose_target_original4, h, w)
            G_input = [source_image, torch.cat((source_pose, output_pose), 1)]
            output_image_generate2 = this_model.generator.forward(G_input, action_label.view(batch_size, ).long())

            pose_source = this_model.get_gaussian_maps(pose_source_original, h, w)
            pose_source = this_model.visual_map(pose_source[0])
            pose_source = torch.Tensor(pose_source).unsqueeze(0).permute(0, 3, 1, 2).cuda()

            pose_target = this_model.get_gaussian_maps(pose_target_original4, h, w)
            pose_target = this_model.visual_map(pose_target[0])
            pose_target = torch.Tensor(pose_target).unsqueeze(0).permute(0, 3, 1, 2).cuda()

            x_img_image.append(output_image_generate2)

            x_save.append(source_image)
            x_target1.append(target_image1)
            x_target2.append(target_image2)
            x_target3.append(target_image3)
            x_target4.append(target_image4)
            x_target5.append(target_image5)

            x_distance1.append(torch.abs(output_image_generate1_1 - output_image_generate2_1))
            x_distance2.append(torch.abs(output_image_generate1_2 - output_image_generate2_2))
            x_distance3.append(torch.abs(output_image_generate1_3 - output_image_generate2_3))
            x_distance4.append(torch.abs(output_image_generate1_4 - output_image_generate2_4))

        x_img_image = torch.cat(x_img_image)
        x_pose_random1_1 = torch.cat(x_pose_random1_1)
        x_img_random1_1 = torch.cat(x_img_random1_1)
        x_pose_random1_2 = torch.cat(x_pose_random1_2)
        x_img_random1_2 = torch.cat(x_img_random1_2)
        x_pose_random1_3 = torch.cat(x_pose_random1_3)
        x_img_random1_3 = torch.cat(x_img_random1_3)
        x_pose_random1_4 = torch.cat(x_pose_random1_4)
        x_img_random1_4 = torch.cat(x_img_random1_4)

        x_pose_random2_1 = torch.cat(x_pose_random2_1)
        x_img_random2_1 = torch.cat(x_img_random2_1)
        x_pose_random2_2 = torch.cat(x_pose_random2_2)
        x_img_random2_2 = torch.cat(x_img_random2_2)
        x_pose_random2_3 = torch.cat(x_pose_random2_3)
        x_img_random2_3 = torch.cat(x_img_random2_3)
        x_pose_random2_4 = torch.cat(x_pose_random2_4)
        x_img_random2_4 = torch.cat(x_img_random2_4)

        x_save = torch.cat(x_save)
        x_target1 = torch.cat(x_target1)
        x_target2 = torch.cat(x_target2)
        x_target3 = torch.cat(x_target3)
        x_target4 = torch.cat(x_target4)
        x_target5 = torch.cat(x_target5)

        x_distance1 = torch.cat(x_distance1)
        x_distance2 = torch.cat(x_distance2)
        x_distance3 = torch.cat(x_distance3)
        x_distance4 = torch.cat(x_distance4)

        self.train()
        return x_save, x_target1, x_target2, x_target3, x_target4, x_target5, \
               x_pose_random1_1, x_img_random1_1, \
               x_pose_random1_2, x_img_random1_2, \
               x_pose_random1_3, x_img_random1_3, \
               x_pose_random1_4, x_img_random1_4, \
               x_distance1, x_distance2, x_distance3, x_distance4, \
               x_pose_random2_1, x_img_random2_1, \
               x_pose_random2_2, x_img_random2_2, \
               x_pose_random2_3, x_img_random2_3, \
               x_pose_random2_4, x_img_random2_4, x_img_image
    
    def update_learning_rate(self):
        if self.classifier_scheduler is not None:
            self.classifier_scheduler.step()
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        print(checkpoint_dir)
        this_model = self.model.module

        last_model_name = get_model_list(checkpoint_dir, "im_encode_content")
        state_dict = torch.load(last_model_name)
        this_model.image_enc_content.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        last_model_name_base = os.path.basename(last_model_name)
        epoch = int(last_model_name_base[0:4])

        last_model_name = get_model_list(checkpoint_dir, "im_encode_vae")
        state_dict = torch.load(last_model_name)
        this_model.image_enc_vae.load_state_dict(state_dict['a'])

        print('VAE from iteration %d' % iterations)
        return iterations, epoch

    def resume2(self, checkpoint_dir, hyperparameters):
        this_model = self.model.module

        last_model_name = get_model_list(checkpoint_dir, "im_encode_content")
        state_dict = torch.load(last_model_name)
        this_model.image_enc_content.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        last_model_name_base = os.path.basename(last_model_name)
        epoch = int(last_model_name_base[0:4])

        last_model_name = get_model_list(checkpoint_dir, "im_encode_vae")
        state_dict = torch.load(last_model_name)
        this_model.image_enc_vae.load_state_dict(state_dict['a'])

        last_model_name = get_model_list(checkpoint_dir, "generator")
        state_dict = torch.load(last_model_name)
        this_model.generator.load_state_dict(state_dict['a'])

        last_model_name = get_model_list(checkpoint_dir, "classifier")
        state_dict = torch.load(last_model_name)
        this_model.classifier.load_state_dict(state_dict['a'])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.generator_opt.load_state_dict(state_dict['generator'])
        self.classifier_opt.load_state_dict(state_dict['classifier'])

        self.generator_scheduler = get_scheduler(self.generator_opt, hyperparameters, iterations)
        self.classifier_scheduler = get_scheduler(self.classifier_opt, hyperparameters, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations, epoch

    def resume_test(self, checkpoint_dir, hyperparameters):
        this_model = self.model.module
        last_model_name = 'pretrain/im_encode_content.pt'
        state_dict = torch.load(last_model_name)
        this_model.image_enc_content.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        last_model_name_base = os.path.basename(last_model_name)
        epoch = int(last_model_name_base[0:4])

        last_model_name = 'pretrain/im_encode_vae.pt'
        state_dict = torch.load(last_model_name)
        this_model.image_enc_vae.load_state_dict(state_dict['a'])

        last_model_name = 'pretrain/generator.pt'
        state_dict = torch.load(last_model_name)
        this_model.generator.load_state_dict(state_dict['a'])

        print('Getting from iteration %d' % iterations)
        return iterations, epoch

    def save(self, snapshot_dir, iterations, epoch):
        # Save generators, discriminators, and optimizers
        this_model = self.model.module
        image_encode_content_name = os.path.join(snapshot_dir, '%04d_im_encode_content_%08d.pt' % (epoch, iterations))
        image_encode_vae_name = os.path.join(snapshot_dir, '%04d_im_encode_vae_%08d.pt' % (epoch, iterations))
        classifier_name = os.path.join(snapshot_dir, '%04d_classifier_%08d.pt' % (epoch, iterations))
        generator_name = os.path.join(snapshot_dir, '%04d_generator_%08d.pt' % (epoch, iterations))

        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': this_model.image_enc_content.state_dict()}, image_encode_content_name)
        torch.save({'a': this_model.image_enc_vae.state_dict()}, image_encode_vae_name)
        torch.save({'a': this_model.classifier.state_dict()}, classifier_name)
        torch.save({'a': this_model.generator.state_dict()}, generator_name)

        torch.save({'classifier': self.classifier_opt.state_dict(),
                    'generator': self.generator_opt.state_dict()}, opt_name)
