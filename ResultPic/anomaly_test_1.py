# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import argparse
import os
import random
from tqdm import tqdm
from common import get_autoencoder, Pdn_small3, Pdn_small2, get_pdn_medium, get_pdn_small,\
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt
import cv2

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='pingshen256nfsq',  #
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/pingshen256nfsq')
    parser.add_argument('-e', '--eval_output_dir', default='eval_output')
    parser.add_argument('-m', '--model_size', default='small',  #
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')  #
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='E:/LN/DataSet/',  #
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=30000)
    return parser.parse_args()


# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

# class EfficientAD(nn.Module):
#     def __init__(self):
#         super(EfficientAD).__init__()
#
#     def forward(self):


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    # output dir
    quantiles_output_dir = os.path.join(config.output_dir, 'quantiles', config.dataset, config.subdataset, 'quantiles')

    # load data

    test_set = ImageFolderWithPath(os.path.join(dataset_path, config.subdataset, 'test'))

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    autoencoder = get_autoencoder(out_channels)

    # load models
    root = config.output_dir + '/trainings/' + config.dataset + '/' + config.subdataset
    teacher_state_dict = torch.load(root + '/teacher_final.pth', map_location='cpu')
    student_state_dict = torch.load(root + '/student_final.pth', map_location='cpu')
    autoencoder_state_dict = torch.load(root + '/autoencoder_final.pth', map_location='cpu')

    teacher.load_state_dict(teacher_state_dict)
    student.load_state_dict(student_state_dict)
    autoencoder.load_state_dict(autoencoder_state_dict)

    # run evaluation
    teacher.eval()
    student.eval()
    autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    quantiles = np.load(quantiles_output_dir + '/{}_quantiles_last.npy'.format(config.subdataset),
                        allow_pickle=True).item()
    teacher_mean = torch.tensor(quantiles['teacher_mean'], device='cuda')
    teacher_std = torch.tensor(quantiles['teacher_std'], device='cuda')
    q_st_start = torch.tensor(quantiles['q_st_start'], device='cuda')
    q_st_end = torch.tensor(quantiles['q_st_end'], device='cuda')
    q_ae_start = torch.tensor(quantiles['q_ae_start'], device='cuda')
    q_ae_end = torch.tensor(quantiles['q_ae_end'], device='cuda')
    good_upper_limit = torch.tensor(quantiles['good_upper_limit'], device='cuda')


    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end, good_upper_limit=good_upper_limit, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, good_upper_limit, desc='Running inference'):
    y_true = []
    y_score = []
    real_class = []
    detect_class = []
    max_scores = []
    thresholds = []
    detect_name = []
    detect_path = []
    original_path = []
    kays = ['实际类型', '检测类型', '得分', '阈值', '检测文件名', '检测文件路径', '原图路径']
    for image, target, path in tqdm(test_set, desc=desc):

        image = default_transform(image)
        C, H, W = image.size()
        img = image.detach().cpu().numpy()
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)

        map_combined = torch.nn.functional.interpolate(map_combined, (H, W), mode='bilinear')  # map_combined：（1，1，64，64）→（1，1，256，256）

        map_combined = map_combined[0, 0].cpu().numpy()  # map_combined：（1，1，256，256）→（256，256）
        map_combined = map_combined.clip(0, 1)

        scores = map_combined * 255
        scores_max = np.max(scores)
        #threshold = good_upper_limit.cpu().numpy() * 255
        threshold =65
        defect_class = os.path.basename(os.path.dirname(path))

        real_class.append(defect_class)
        detect_name.append(defect_class + '_{}'.format(os.path.split(path)[1].split('.')[0]))
        max_scores.append('%.1f'%scores_max)
        thresholds.append('%.1f'%threshold)
        original_path.append('=HYPERLINK("{}", "打开原图")'.format(path))

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        vmin = 0    # np.min(map_combined) * 255.
        vmax = 255  #np.max(map_combined) * 255.


        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)  # denormalize
        heat_map = scores

        mask = scores.copy()
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        mask *= 255

        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        # img、heat_map、mask、vis_img的格式为ndarray, 长宽调整为356、宽调节为292
        img = cv2.resize(img, (356, 292))
        heat_map = cv2.resize(heat_map, (356, 292))
        mask = cv2.resize(mask, (356, 292))
        vis_img = cv2.resize(vis_img, (356, 292))


        fig_img, ax_img = plt.subplots(1, 5, figsize=(24, 6))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text(os.path.basename(path))

        ax_img[1].imshow(heat_map, cmap='jet', alpha=1, interpolation='none', norm=norm)
        ax_img[1].title.set_text('Max_score=' + str('%.1f' % scores_max))
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)

        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none',norm=norm)
        ax_img[2].title.set_text(str('T_lim=' + str('%.1f' % threshold)))

        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')

        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        # 显示右侧颜色条
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        defect_class = os.path.basename(os.path.dirname(path))  # 缺陷类型

        cb.set_label('Anomaly Score', fontdict=font)

        path_root = 'E:\\LN\\ModelResult\\pingshen\\'
        WuDir = os.path.join(path_root, 'WuJian-MaxScore')
        DefectDir = os.path.join(path_root, 'Defect')
        GoodDir = os.path.join(path_root, 'Good')
        if not os.path.isdir(WuDir):
            os.makedirs(WuDir)
        if not os.path.isdir(DefectDir):
            os.makedirs(DefectDir)
        if not os.path.isdir(GoodDir):
            os.makedirs(GoodDir)

        if scores.max() < threshold:
            fig_img.savefig(os.path.join(GoodDir, defect_class + '_{}.jpg'.format(os.path.split(path)[1].split('.')[0])),dpi=100)
            detect_class.append('good')
            detect_path.append('=HYPERLINK("{}", "打开检测图")'.format((os.path.join(GoodDir, defect_class + '_{}'+'.jpg').format(os.path.split(path)[1].split('.')[0]))))
        else:
            fig_img.savefig(os.path.join(DefectDir, defect_class + '_{}.jpg'.format(os.path.split(path)[1].split('.')[0])), dpi=100)
            detect_class.append('bad')
            detect_path.append('=HYPERLINK("{}", "打开检测图")'.format((os.path.join(DefectDir, defect_class + '_{}'+ '.jpg').format(os.path.split(path)[1].split('.')[0]))))
        plt.close()

    data = [real_class, detect_class, max_scores, thresholds, detect_name, detect_path, original_path]
    if len(kays) == len(data):
        # 使用字典推导式方法将两个元组都转换为字典
        # 使用enumerate()函数
        resultDictionary = {kays[i]: data[i] for i, _ in enumerate(data)}
    data_file = pd.DataFrame(resultDictionary, index=range(len(real_class)), columns=kays)
    data_file.to_csv(os.path.join(path_root, 'out' + '.csv'))

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:]) ** 2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae


if __name__ == '__main__':
    main()
