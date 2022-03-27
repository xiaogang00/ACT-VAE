import sys
import cv2
import math
import torch
import random
import numpy as np
import torch.utils.data as data
sys.path.append('.')
import utils

from utils.transforms import letterbox
sys.modules['utils'] = utils

cv2.setNumThreads(0)


def bbox_randscale(bbox, miniou=0.75):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    scale_shrink = (1-math.sqrt(miniou))/2.
    scale_expand = (math.sqrt(1./miniou)-1)/2.
    w1,h1 = random.uniform(-scale_expand, scale_shrink)*w, random.uniform(-scale_expand, scale_shrink)*h
    w2,h2 = random.uniform(-scale_shrink, scale_expand)*w, random.uniform(-scale_shrink, scale_expand)*h
    bbox[0],bbox[2] = bbox[0]+w1,bbox[2]+w2
    bbox[1],bbox[3] = bbox[1]+h1,bbox[3]+h2
    return bbox


class ReferDataset(data.Dataset):
    def __init__(self, data_root, data_list, imsize=256, transform=None, augment=False, testmode=False,
                 split='train'):
        self.images = []
        self.data_root = data_root
        self.imsize = imsize
        self.transform = transform

        self.testmode = testmode
        self.split = split
        self.augment=augment

        f = open(data_list)
        lines = f.readlines()
        self.image_list = []
        for mm in range(len(lines)):
            this_line = lines[mm].strip()
            this_line_split = this_line.split(',')
            source_image_name = this_line_split[0]
            target_image_name1 = this_line_split[1]
            target_image_name2 = this_line_split[2]
            target_image_name3 = this_line_split[3]
            target_image_name4 = this_line_split[4]
            target_image_name5 = this_line_split[5]

            action_name = this_line_split[6]
            action_id = int(this_line_split[7])

            bbox_x1 = int(this_line_split[8])
            bbox_y1 = int(this_line_split[9])
            bbox_x2 = int(this_line_split[10])
            bbox_y2 = int(this_line_split[11])
            bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]

            mask_x1 = int(this_line_split[12])
            mask_y1 = int(this_line_split[13])
            mask_x2 = int(this_line_split[14])
            mask_y2 = int(this_line_split[15])
            mask1 = [mask_x1, mask_y1, mask_x2, mask_y2]

            mask_x1 = int(this_line_split[16])
            mask_y1 = int(this_line_split[17])
            mask_x2 = int(this_line_split[18])
            mask_y2 = int(this_line_split[19])
            mask2 = [mask_x1, mask_y1, mask_x2, mask_y2]

            mask_x1 = int(this_line_split[20])
            mask_y1 = int(this_line_split[21])
            mask_x2 = int(this_line_split[22])
            mask_y2 = int(this_line_split[23])
            mask3 = [mask_x1, mask_y1, mask_x2, mask_y2]

            mask_x1 = int(this_line_split[24])
            mask_y1 = int(this_line_split[25])
            mask_x2 = int(this_line_split[26])
            mask_y2 = int(this_line_split[27])
            mask4 = [mask_x1, mask_y1, mask_x2, mask_y2]

            mask_x1 = int(this_line_split[28])
            mask_y1 = int(this_line_split[29])
            mask_x2 = int(this_line_split[30])
            mask_y2 = int(this_line_split[31])
            mask5 = [mask_x1, mask_y1, mask_x2, mask_y2]

            point_target1 = []
            for oo in range(13):
                point_x = float(this_line_split[31 + oo*2 + 1])
                point_y = float(this_line_split[31 + oo*2 + 2])
                point_target1.append([point_x, point_y])
            point_target1 = np.array(point_target1)

            point_target2 = []
            for oo in range(13):
                point_x = float(this_line_split[57 + oo*2 + 1])
                point_y = float(this_line_split[57 + oo*2 + 2])
                point_target2.append([point_x, point_y])
            point_target2 = np.array(point_target2)

            point_target3 = []
            for oo in range(13):
                point_x = float(this_line_split[83 + oo*2 + 1])
                point_y = float(this_line_split[83 + oo*2 + 2])
                point_target3.append([point_x, point_y])
            point_target3 = np.array(point_target3)

            point_target4 = []
            for oo in range(13):
                point_x = float(this_line_split[109 + oo*2 + 1])
                point_y = float(this_line_split[109 + oo*2 + 2])
                point_target4.append([point_x, point_y])
            point_target4 = np.array(point_target4)

            point_target5 = []
            for oo in range(13):
                point_x = float(this_line_split[135 + oo*2 + 1])
                point_y = float(this_line_split[135 + oo*2 + 2])
                point_target5.append([point_x, point_y])
            point_target5 = np.array(point_target5)

            point_source = []
            for oo in range(13):
                point_x = float(this_line_split[161 + oo * 2 + 1])
                point_y = float(this_line_split[161 + oo * 2 + 2])
                point_source.append([point_x, point_y])
            point_source = np.array(point_source)

            self.image_list.append([source_image_name, 
                                    target_image_name1, target_image_name2, target_image_name3,
                                    target_image_name4, target_image_name5,
                                    action_id, bbox, 
                                    mask1, mask2, mask3, mask4, mask5, point_source, 
                                    point_target1, point_target2, point_target3, point_target4, point_target5])

    def pull_item(self, img_file):
        img_file = img_file.replace('/mnt/backup/project/xgxu/landmark_data/Penn_Action_new/frames', '/mnt/proj3/xgxu/action_condition_synthesis/data/Penn_frames')
        img_path = img_file
        img = cv2.imread(img_path)
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        idx = idx % (len(self.image_list))
        source_img_name, target_img_name1, target_img_name2, target_img_name3, \
        target_img_name4, target_img_name5, action_id, bbox, \
        mask1, mask2, mask3, mask4, mask5, \
        point_source_origin, \
        point_target1_origin, point_target2_origin, point_target3_origin, \
        point_target4_origin, point_target5_origin = self.image_list[idx]

        source_img = self.pull_item(source_img_name)
        target_img1 = self.pull_item(target_img_name1)
        target_img2 = self.pull_item(target_img_name2)
        target_img3 = self.pull_item(target_img_name3)
        target_img4 = self.pull_item(target_img_name4)
        target_img5 = self.pull_item(target_img_name5)
        point_source = point_source_origin.copy()
        point_target1 = point_target1_origin.copy()
        point_target2 = point_target2_origin.copy()
        point_target3 = point_target3_origin.copy()
        point_target4 = point_target4_origin.copy()
        point_target5 = point_target5_origin.copy()

        mask_img1 = np.zeros_like(source_img)
        h = mask_img1.shape[0]
        w = mask_img1.shape[1]
        x1 = max(0, mask1[0])
        y1 = max(0, mask1[1])
        x2 = min(w-1, mask1[2])
        y2 = min(h-1, mask1[3])
        mask_img1[y1:y2, x1:x2, :] = 1

        mask_img2 = np.zeros_like(source_img)
        h = mask_img2.shape[0]
        w = mask_img2.shape[1]
        x1 = max(0, mask2[0])
        y1 = max(0, mask2[1])
        x2 = min(w-1, mask2[2])
        y2 = min(h-1, mask2[3])
        mask_img2[y1:y2, x1:x2, :] = 1

        mask_img3 = np.zeros_like(source_img)
        h = mask_img3.shape[0]
        w = mask_img3.shape[1]
        x1 = max(0, mask3[0])
        y1 = max(0, mask3[1])
        x2 = min(w-1, mask3[2])
        y2 = min(h-1, mask3[3])
        mask_img3[y1:y2, x1:x2, :] = 1

        mask_img4 = np.zeros_like(source_img)
        h = mask_img4.shape[0]
        w = mask_img4.shape[1]
        x1 = max(0, mask4[0])
        y1 = max(0, mask4[1])
        x2 = min(w-1, mask4[2])
        y2 = min(h-1, mask4[3])
        mask_img4[y1:y2, x1:x2, :] = 1

        mask_img5 = np.zeros_like(source_img)
        h = mask_img5.shape[0]
        w = mask_img5.shape[1]
        x1 = max(0, mask5[0])
        y1 = max(0, mask5[1])
        x2 = min(w-1, mask5[2])
        y2 = min(h-1, mask5[3])
        mask_img5[y1:y2, x1:x2, :] = 1

        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w - 1, bbox[2])
        y2 = min(h - 1, bbox[3])
        source_img = source_img[y1:y2, x1:x2, :]
        target_img1 = target_img1[y1:y2, x1:x2, :]
        target_img2 = target_img2[y1:y2, x1:x2, :]
        target_img3 = target_img3[y1:y2, x1:x2, :]
        target_img4 = target_img4[y1:y2, x1:x2, :]
        target_img5 = target_img5[y1:y2, x1:x2, :]
        mask_img1 = mask_img1[y1:y2, x1:x2, :]
        mask_img2 = mask_img2[y1:y2, x1:x2, :]
        mask_img3 = mask_img3[y1:y2, x1:x2, :]
        mask_img4 = mask_img4[y1:y2, x1:x2, :]
        mask_img5 = mask_img5[y1:y2, x1:x2, :]

        point_source[:, 0] = point_source[:, 0] - x1
        point_source[:, 1] = point_source[:, 1] - y1
        point_target1[:, 0] = point_target1[:, 0] - x1
        point_target1[:, 1] = point_target1[:, 1] - y1
        point_target2[:, 0] = point_target2[:, 0] - x1
        point_target2[:, 1] = point_target2[:, 1] - y1
        point_target3[:, 0] = point_target3[:, 0] - x1
        point_target3[:, 1] = point_target3[:, 1] - y1
        point_target4[:, 0] = point_target4[:, 0] - x1
        point_target4[:, 1] = point_target4[:, 1] - y1
        point_target5[:, 0] = point_target5[:, 0] - x1
        point_target5[:, 1] = point_target5[:, 1] - y1
        
        source_img, _, ratio, dw, dh = letterbox(source_img, None, self.imsize)
        target_img1, mask_img1, ratio, dw, dh = letterbox(target_img1, mask_img1, self.imsize)
        target_img2, mask_img2, ratio, dw, dh = letterbox(target_img2, mask_img2, self.imsize)
        target_img3, mask_img3, ratio, dw, dh = letterbox(target_img3, mask_img3, self.imsize)
        target_img4, mask_img4, ratio, dw, dh = letterbox(target_img4, mask_img4, self.imsize)
        target_img5, mask_img5, ratio, dw, dh = letterbox(target_img5, mask_img5, self.imsize)
        point_source[:, 0] = point_source[:, 0] * ratio + dw
        point_source[:, 1] = point_source[:, 1] * ratio + dh
        point_target1[:, 0] = point_target1[:, 0] * ratio + dw
        point_target1[:, 1] = point_target1[:, 1] * ratio + dh
        point_target2[:, 0] = point_target2[:, 0] * ratio + dw
        point_target2[:, 1] = point_target2[:, 1] * ratio + dh
        point_target3[:, 0] = point_target3[:, 0] * ratio + dw
        point_target3[:, 1] = point_target3[:, 1] * ratio + dh
        point_target4[:, 0] = point_target4[:, 0] * ratio + dw
        point_target4[:, 1] = point_target4[:, 1] * ratio + dh
        point_target5[:, 0] = point_target5[:, 0] * ratio + dw
        point_target5[:, 1] = point_target5[:, 1] * ratio + dh

        mask_img1 = torch.Tensor(mask_img1).float()
        mask_img1 = mask_img1.permute(2, 0, 1)
        mask_img2 = torch.Tensor(mask_img2).float()
        mask_img2 = mask_img2.permute(2, 0, 1)
        mask_img3 = torch.Tensor(mask_img3).float()
        mask_img3 = mask_img3.permute(2, 0, 1)
        mask_img4 = torch.Tensor(mask_img4).float()
        mask_img4 = mask_img4.permute(2, 0, 1)
        mask_img5 = torch.Tensor(mask_img5).float()
        mask_img5 = mask_img5.permute(2, 0, 1)

        point_source = np.clip(point_source, 0, self.imsize - 1)
        point_target1 = np.clip(point_target1, 0, self.imsize - 1)
        point_target2 = np.clip(point_target2, 0, self.imsize - 1)
        point_target3 = np.clip(point_target3, 0, self.imsize - 1)
        point_target4 = np.clip(point_target4, 0, self.imsize - 1)
        point_target5 = np.clip(point_target5, 0, self.imsize - 1)

        point_source = 2 * (point_source * 1.0 / self.imsize) - 1
        point_target1 = 2 * (point_target1 * 1.0 / self.imsize) - 1
        point_target2 = 2 * (point_target2 * 1.0 / self.imsize) - 1
        point_target3 = 2 * (point_target3 * 1.0 / self.imsize) - 1
        point_target4 = 2 * (point_target4 * 1.0 / self.imsize) - 1
        point_target5 = 2 * (point_target5 * 1.0 / self.imsize) - 1

        point_source = torch.Tensor(point_source).float()
        point_target1 = torch.Tensor(point_target1).float()
        point_target2 = torch.Tensor(point_target2).float()
        point_target3 = torch.Tensor(point_target3).float()
        point_target4 = torch.Tensor(point_target4).float()
        point_target5 = torch.Tensor(point_target5).float()

        if self.transform is not None:
            source_img = self.transform(source_img)
            target_img1 = self.transform(target_img1)
            target_img2 = self.transform(target_img2)
            target_img3 = self.transform(target_img3)
            target_img4 = self.transform(target_img4)
            target_img5 = self.transform(target_img5)

        action_class = [action_id]
        action_class = torch.Tensor(action_class)

        return source_img, target_img1, target_img2, target_img3, target_img4, target_img5, \
               mask_img1, mask_img2, mask_img3, mask_img4, mask_img5, \
               action_class, point_source, \
               point_target1, point_target2, point_target3, point_target4, point_target5

