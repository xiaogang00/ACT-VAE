import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
try:
    from itertools import izip as zip
except ImportError:
    pass
import tensorboardX
import shutil
import imageio
import time
import os

from utils_file import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
from trainer_actvae import Trainer
from dataset.image_loader2 import *
from utils.utils import *


def writeTensor(save_path, tensor, nRow=16, row_first=False):
    nSample = tensor.shape[0]
    nCol = np.int16(nSample / nRow)
    all = []
    k = 0
    for iCol in range(nCol):
        all_ = []
        for iRow in range(nRow):
            now = tensor[k, :, :, :]
            now = now.permute(1, 2, 0)
            all_ += [now]
            k += 1
        if not row_first:
            all += [torch.cat(all_, dim=0)]
        else:
            all += [torch.cat(all_, dim=1)]
    if not row_first:
        all = torch.cat(all, dim=1)
    else:
        all = torch.cat(all, dim=0)
    all = all.cpu().numpy().astype(np.uint8)
    print('saving tensor to %s' % save_path)
    imageio.imwrite(save_path, all)


def untransformTensor(vggImageTensor):
    vggImageTensor = vggImageTensor.cpu()
    vggImageTensor = (vggImageTensor + 1) / 2
    vggImageTensor.clamp_(0, 1)

    vggImageTensor = vggImageTensor.numpy()
    vggImageTensor[vggImageTensor > 1.] = 1.
    vggImageTensor[vggImageTensor < 0.] = 0.
    vggImageTensor = vggImageTensor * 255
    return vggImageTensor


def generate_random_sequence(length, number):
    random_list = []
    for mm in range(number):
        random_index = random.randint(0, length-1)
        random_list.append(random_index)
    return random_list


def train(config, opts):
    train_epoch = config['train_epoch']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    data_root = config['data_root']
    train_data_list = config['train_data_list']
    test_data_list = config['test_data_list']
    imsize = config['imsize']
    batch_size = config['batch_size']
    workers = config['workers']

    input_transform_generation = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_dataset = ReferDataset(data_root=data_root,
                                 data_list=train_data_list,
                                 split='trainval',
                                 imsize=imsize,
                                 transform=input_transform_generation,
                                 augment=False)

    test_dataset = ReferDataset(data_root=data_root,
                                data_list=test_data_list,
                                testmode=True,
                                split='testA',
                                imsize=imsize,
                                transform=input_transform_generation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=0)
    ngpus = torch.cuda.device_count()
    print("Number of GPUs: %d" % ngpus)

    trainer = Trainer(config)
    trainer.cuda()
    trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus))

    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = model_name + opts.name

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    if opts.resume:
        iterations, start_epoch = trainer.resume(checkpoint_directory, hyperparameters=config)
    else:
        iterations, start_epoch = 0, 0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_KL = AverageMeter()
    loss_recon_pose = AverageMeter()

    end = time.time()
    for epoch in range(start_epoch, train_epoch, 1):
        for batch_idx, (source_image, target_image1, target_image2, target_image3, target_image4, target_image5,
                        mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_id, pose_source, 
                        pose_target1, pose_target2, pose_target3, pose_target4, pose_target5) in enumerate(train_loader):
            source_image = source_image.cuda().float()
            target_image1 = target_image1.cuda().float()
            target_image2 = target_image2.cuda().float()
            target_image3 = target_image3.cuda().float()
            target_image4 = target_image4.cuda().float()
            target_image5 = target_image5.cuda().float()
            mask_image1 = mask_image1.cuda().float()
            mask_image2 = mask_image2.cuda().float()
            mask_image3 = mask_image3.cuda().float()
            mask_image4 = mask_image4.cuda().float()
            mask_image5 = mask_image5.cuda().float()
            action_id = action_id.cuda().float()
            pose_source = pose_source.cuda().float()
            pose_target1 = pose_target1.cuda().float()
            pose_target2 = pose_target2.cuda().float()
            pose_target3 = pose_target3.cuda().float()
            pose_target4 = pose_target4.cuda().float()
            pose_target5 = pose_target5.cuda().float()

            source_image = Variable(source_image)
            target_image1 = Variable(target_image1)
            target_image2 = Variable(target_image2)
            target_image3 = Variable(target_image3)
            target_image4 = Variable(target_image4)
            target_image5 = Variable(target_image5)
            mask_image1 = Variable(mask_image1)
            mask_image2 = Variable(mask_image2)
            mask_image3 = Variable(mask_image3)
            mask_image4 = Variable(mask_image4)
            mask_image5 = Variable(mask_image5)
            action_id = Variable(action_id)
            pose_source = Variable(pose_source)
            pose_target1 = Variable(pose_target1)
            pose_target2 = Variable(pose_target2)
            pose_target3 = Variable(pose_target3)
            pose_target4 = Variable(pose_target4)
            pose_target5 = Variable(pose_target5)

            trainer.update_learning_rate()
            with Timer("Elapsed time in update: %f"):
                trainer.gen_update(source_image, target_image1, target_image2, target_image3, target_image4, target_image5, 
                                   mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_id, 
                                   pose_source, pose_target1, pose_target2, pose_target3, pose_target4, pose_target5)
                torch.cuda.synchronize()
            loss_recon_pose.update(trainer.loss_recon_pose.cpu().item(), source_image.size(0))
            loss_KL.update(trainer.loss_KL.cpu().item(), source_image.size(0))

            if (iterations + 1) % config['log_iter'] == 0:
                print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t' \
                            'Loss_recon_pose {loss_recon_pose.val:.4f} ({loss_recon_pose.avg:.4f})\t'.format(epoch, batch_idx, len(train_loader), batch_time=batch_time,
                                                                                                             data_time=data_time, loss_KL=loss_KL, loss_recon_pose=loss_recon_pose)
                print(print_str)
                write_loss(iterations, trainer, train_writer)

            list_train_a = generate_random_sequence(len(train_loader.dataset), display_size)
            list_test_a = generate_random_sequence(len(test_loader.dataset), display_size)

            #####################################################################################################
            train_dis_all = [train_loader.dataset[i] for i in list_train_a]
            test_dis_all = [test_loader.dataset[i] for i in list_test_a]

            train_dis_im_source = torch.cat([image[0].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_source = torch.cat([image[0].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()

            train_dis_im_target1 = torch.cat([image[1].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_target1 = torch.cat([image[1].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_target2 = torch.cat([image[2].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_target2 = torch.cat([image[2].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_target3 = torch.cat([image[3].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_target3 = torch.cat([image[3].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_target4 = torch.cat([image[4].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_target4 = torch.cat([image[4].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_target5 = torch.cat([image[5].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_target5 = torch.cat([image[5].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()

            train_dis_im_mask1 = torch.cat([image[6].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_mask1 = torch.cat([image[6].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_mask2 = torch.cat([image[7].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_mask2 = torch.cat([image[7].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_mask3 = torch.cat([image[8].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_mask3 = torch.cat([image[8].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_mask4 = torch.cat([image[9].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_mask4 = torch.cat([image[9].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_im_mask5 = torch.cat([image[10].unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_im_mask5 = torch.cat([image[10].unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()

            train_dis_id = torch.cat([torch.Tensor(image[11]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_id = torch.cat([torch.Tensor(image[11]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()

            train_dis_pose_source = torch.cat([torch.Tensor(image[12]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_source = torch.cat([torch.Tensor(image[12]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_pose_target1 = torch.cat([torch.Tensor(image[13]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_target1 = torch.cat([torch.Tensor(image[13]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_pose_target2 = torch.cat([torch.Tensor(image[14]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_target2 = torch.cat([torch.Tensor(image[14]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_pose_target3 = torch.cat([torch.Tensor(image[15]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_target3 = torch.cat([torch.Tensor(image[15]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_pose_target4 = torch.cat([torch.Tensor(image[16]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_target4 = torch.cat([torch.Tensor(image[16]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            train_dis_pose_target5 = torch.cat([torch.Tensor(image[17]).unsqueeze(0).float() for image in train_dis_all], dim=0).cuda()
            test_dis_pose_target5 = torch.cat([torch.Tensor(image[17]).unsqueeze(0).float() for image in test_dis_all], dim=0).cuda()
            #####################################################################################################

            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(train_dis_im_source, train_dis_im_target1,
                                                        train_dis_im_target2, train_dis_im_target3,
                                                        train_dis_im_target4, train_dis_im_target5,
                                                        train_dis_im_mask1, train_dis_im_mask2,
                                                        train_dis_im_mask3, train_dis_im_mask4,
                                                        train_dis_im_mask5, train_dis_id,
                                                        train_dis_pose_source, train_dis_pose_target1,
                                                        train_dis_pose_target2, train_dis_pose_target3,
                                                        train_dis_pose_target4, train_dis_pose_target5)
                    train_image_outputs = trainer.sample(test_dis_im_source, test_dis_im_target1,
                                                         test_dis_im_target2, test_dis_im_target3,
                                                         test_dis_im_target4, test_dis_im_target5,
                                                         test_dis_im_mask1, test_dis_im_mask2,
                                                         test_dis_im_mask3, test_dis_im_mask4,
                                                         test_dis_im_mask5, test_dis_id,
                                                         test_dis_pose_source, test_dis_pose_target1,
                                                         test_dis_pose_target2, test_dis_pose_target3,
                                                         test_dis_pose_target4, test_dis_pose_target5)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (epoch))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (epoch))
                write_html(output_directory + "/index.html", epoch, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    train_outputs = trainer.sample(train_dis_im_source, train_dis_im_target1,
                                                   train_dis_im_target2, train_dis_im_target3,
                                                   train_dis_im_target4, train_dis_im_target5,
                                                   train_dis_im_mask1, train_dis_im_mask2,
                                                   train_dis_im_mask3, train_dis_im_mask4,
                                                   train_dis_im_mask5, train_dis_id,
                                                   train_dis_pose_source, train_dis_pose_target1,
                                                   train_dis_pose_target2, train_dis_pose_target3,
                                                   train_dis_pose_target4, train_dis_pose_target5)
                write_2images(train_outputs, display_size, image_directory, 'train_current')

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations, epoch)
            iterations += 1
            batch_time.update(time.time() - end)
            end = time.time()


def test2(config, opts):
    from dataset.image_loader2_test import ReferDataset_test

    config['vgg_model_path'] = opts.output_path
    config['batch_size'] = 1

    input_transform_generation = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    data_root = config['data_root']
    test_data_list = config['test_data_list']
    imsize = config['imsize']

    test_dataset = ReferDataset_test(data_root=data_root,
                                     data_list=test_data_list,
                                     testmode=True,
                                     split='testA',
                                     imsize=imsize,
                                     transform=input_transform_generation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                             pin_memory=True, drop_last=True, num_workers=0)
    ngpus = torch.cuda.device_count()
    print("Number of GPUs: %d" % ngpus)

    trainer = Trainer(config)
    trainer.cuda()
    trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus))

    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = model_name + opts.name

    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    iterations, start_epoch = trainer.resume_test(checkpoint_directory, hyperparameters=config)

    output_im_real = 'outputs/Penn_actvae/results_image_real'
    output_im_fake = 'outputs/Penn_actvae/results_image_fake'
    output_gif_real = 'outputs/Penn_actvae/results_gif_real'
    output_gif_fake = 'outputs/Penn_actvae/results_gif_fake'
    if not(os.path.exists(output_im_real)):
        os.mkdir(output_im_real)
    if not(os.path.exists(output_im_fake)):
        os.mkdir(output_im_fake)
    if not(os.path.exists(output_gif_real)):
        os.mkdir(output_gif_real)
    if not(os.path.exists(output_gif_fake)):
        os.mkdir(output_gif_fake)

    count = 0
    for batch_idx, (source_image, target_image1, target_image2, target_image3, target_image4, target_image5,
                    mask_image1, mask_image2, mask_image3, mask_image4, mask_image5, action_id, pose_source,
                    pose_target1, pose_target2, pose_target3, pose_target4, pose_target5, seq_id_all, image_id_all, gap_all) in enumerate(test_loader):
        source_image = source_image.cuda().float()
        target_image1 = target_image1.cuda().float()
        target_image2 = target_image2.cuda().float()
        target_image3 = target_image3.cuda().float()
        target_image4 = target_image4.cuda().float()
        target_image5 = target_image5.cuda().float()
        mask_image1 = mask_image1.cuda().float()
        mask_image2 = mask_image2.cuda().float()
        mask_image3 = mask_image3.cuda().float()
        mask_image4 = mask_image4.cuda().float()
        mask_image5 = mask_image5.cuda().float()
        action_id = action_id.cuda().float()
        pose_source = pose_source.cuda().float()
        pose_target1 = pose_target1.cuda().float()
        pose_target2 = pose_target2.cuda().float()
        pose_target3 = pose_target3.cuda().float()
        pose_target4 = pose_target4.cuda().float()
        pose_target5 = pose_target5.cuda().float()

        source_image = Variable(source_image)
        target_image1 = Variable(target_image1)
        target_image2 = Variable(target_image2)
        target_image3 = Variable(target_image3)
        target_image4 = Variable(target_image4)
        target_image5 = Variable(target_image5)
        mask_image1 = Variable(mask_image1)
        mask_image2 = Variable(mask_image2)
        mask_image3 = Variable(mask_image3)
        mask_image4 = Variable(mask_image4)
        mask_image5 = Variable(mask_image5)
        action_id = Variable(action_id)
        pose_source = Variable(pose_source)
        pose_target1 = Variable(pose_target1)
        pose_target2 = Variable(pose_target2)
        pose_target3 = Variable(pose_target3)
        pose_target4 = Variable(pose_target4)
        pose_target5 = Variable(pose_target5)

        ## first time for generation
        with Timer("Elapsed time in update: %f"):
            pose_seq, pose_seq_real = trainer.test_forward3(source_image, target_image1, target_image2,
                                                            target_image3, target_image4, target_image5,
                                                            mask_image1, mask_image2, mask_image3, mask_image4,
                                                            mask_image5, action_id,
                                                            pose_source, pose_target1, pose_target2, pose_target3,
                                                            pose_target4, pose_target5)
        batch_size = pose_seq.shape[0]
        seq_id_all = seq_id_all.view(1, ).detach().numpy()
        image_id_all = image_id_all.view(1, ).detach().numpy()
        seq_id = int(seq_id_all[0])
        image_id = int(image_id_all[0])

        img_collection_real = []
        img_collection_fake = []
        for mm in range(batch_size):
            pose_frame = untransformTensor(pose_seq[mm].data.cpu())
            pose_frame = np.moveaxis(pose_frame, 0, 2).astype(np.uint8).copy()

            pose_frame_real = untransformTensor(pose_seq_real[mm].data.cpu())
            pose_frame_real = np.moveaxis(pose_frame_real, 0, 2).astype(np.uint8).copy()

            if not(mm==0):
                img_collection_real += [pose_frame_real]
                img_collection_fake += [pose_frame]
        
        imageio.mimsave(os.path.join(output_gif_real, '%04d' % seq_id + '_' + '%04d.gif' % (image_id)), img_collection_real, fps=2)
        imageio.mimsave(os.path.join(output_gif_fake, '%04d' % seq_id + '_' + '%04d_0.gif' % (image_id)), img_collection_fake, fps=2)


        ## second time for generation
        with Timer("Elapsed time in update: %f"):
            pose_seq, pose_seq_real = trainer.test_forward3(source_image, target_image1, target_image2,
                                                            target_image3, target_image4, target_image5,
                                                            mask_image1, mask_image2, mask_image3, mask_image4,
                                                            mask_image5, action_id,
                                                            pose_source, pose_target1, pose_target2, pose_target3,
                                                            pose_target4, pose_target5)
        img_collection_real2 = []
        img_collection_fake2 = []
        for mm in range(batch_size):
            pose_frame = untransformTensor(pose_seq[mm].data.cpu())
            pose_frame = np.moveaxis(pose_frame, 0, 2).astype(np.uint8).copy()

            pose_frame_real = untransformTensor(pose_seq_real[mm].data.cpu())
            pose_frame_real = np.moveaxis(pose_frame_real, 0, 2).astype(np.uint8).copy()

            if not(mm==0):
                img_collection_real2 += [pose_frame_real]
                img_collection_fake2 += [pose_frame]
        imageio.mimsave(os.path.join(output_gif_fake, '%04d' % seq_id + '_' + '%04d_1.gif' % (image_id)), img_collection_fake, fps=2)

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml',
                        help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    opts = parser.parse_args()

    cudnn.benchmark = True

    config = get_config(opts.config)

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(opts.seed)
    np.random.seed(opts.seed + 1)
    torch.manual_seed(opts.seed + 2)
    torch.cuda.manual_seed_all(opts.seed + 3)

    if opts.test:
        test2(config, opts)
    else:
        train(config, opts)

