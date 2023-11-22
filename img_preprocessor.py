# File heavily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Literal
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn

from ladi_vton.src.utils.posemap import get_coco_body25_mapping
from ladi_vton.src.utils.posemap import kpoint_to_heatmap

from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

class ImgPreprocessor():
    def __init__(self,
                 radius=5,
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'captions', 'category', 'warped_cloth', 'clip_cloth_features'),
                 size: Tuple[int, int] = (512, 384),
                 weight_dtype=torch.float16
                 ):

        self.category = ('upper_body')
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.weight_dtype = weight_dtype
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        possible_outputs = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton',
                            'im_mask', 'inpaint_mask', 'parse_mask_total', 'captions',
                            'category', 'warped_cloth', 'clip_cloth_features']

        assert all(x in possible_outputs for x in outputlist)

        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                     dataset="vitonhd")
        self.tps.eval()
        self.refinement.eval()

        # if "clip_cloth_features" in self.outputlist:
        #     self.clip_cloth_features = torch.load(os.path.join(
        #         PROJECT_ROOT / 'data', 'clip_cloth_embeddings', 'vitonhd', f'{phase}_last_hidden_state_features.pt'),
        #         map_location='cpu').detach().requires_grad_(False)

        #     with open(os.path.join(
        #             PROJECT_ROOT / 'data', 'clip_cloth_embeddings', 'vitonhd', f'{phase}_features_names.pkl'), 'rb') as f:
        #         self.clip_cloth_features_names = pickle.load(f)

        # for segm
        # self.semantic_nc = 13
        # self.load_height = 1024
        # self.load_width = 768

        # self.seg = SegGenerator(input_nc=self.semantic_nc + 8, output_nc=self.semantic_nc)
        # self.seg.load_state_dict(torch.load("checkpoints/seg_final.pth"))
        # self.up = nn.Upsample(size=(self.load_height, self.load_width), mode='bilinear')
        # self.gauss = T.GaussianBlur((15, 15), (3, 3))
        # self.gauss.cuda()


    def preprocess(self, person_image, cloth_image):
        category = 'upper_body'

        if "clip_cloth_features" in self.outputlist:  # Precomputed CLIP in-shop embeddings
            clip_cloth_features = None

        if "cloth" in self.outputlist:  # In-shop clothing image
            # Clothing image
            cloth = cloth_image
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            # Person image
            image = person_image
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]


        labels = {
            0: ['background', [0, 10]],  # 0 is background, 10 is neck
            1: ['hair', [1, 2]],  # 1 and 2 are hair
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist or "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist or "pose_map" in self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist:
            # Label Map
            im_parse = self.get_image_parse(image)
            im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 9).astype(np.float32) + \
                                (parse_array == 10).astype(np.float32) + \
                                (parse_array == 8).astype(np.float32)

            parser_mask_changeable = (parse_array == 0).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            parse_cloth = (parse_array == 4).astype(np.float32) + \
                          (parse_array == 7).astype(np.float32) + \
                          (parse_array == 17).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32) + \
                         (parse_array == 7).astype(np.float32) + \
                         (parse_array == 17).astype(np.float32)

            parser_mask_fixed = parser_mask_fixed + (parse_array == 6).astype(np.float32) + \
                                (parse_array == 12).astype(np.float32)  # the lower body is fixed

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            # dilation
            parse_mask = parse_mask.cpu().numpy()

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)

            # Load pose points
            pose_data = self.get_image_keypoints(image)
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
            pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

            pose_mapping = get_coco_body25_mapping()

            # point_num = pose_data.shape[0]
            point_num = len(pose_mapping)

            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)

                point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
                point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []

            for idx in range(point_num):
                ux = pose_data[pose_mapping[idx], 0]  # / (192)
                uy = (pose_data[pose_mapping[idx], 1])  # / (256)

                # scale posemap points
                px = ux  # * self.width
                py = uy  # * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)

            # do in any case because i have only upperbody
            data = self.get_image_keypoints(image)
            data = np.array(data)
            data = data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            data[:, 0] = data[:, 0] * (self.width / 768)
            data[:, 1] = data[:, 1] * (self.height / 1024)

            shoulder_right = tuple(data[pose_mapping[2]])
            shoulder_left = tuple(data[pose_mapping[5]])
            elbow_right = tuple(data[pose_mapping[3]])
            elbow_left = tuple(data[pose_mapping[6]])
            wrist_right = tuple(data[pose_mapping[4]])
            wrist_left = tuple(data[pose_mapping[7]])

            ARM_LINE_WIDTH = int(90 / 512 * self.height)
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            # tune the amount of dilation here
            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total
            inpaint_mask = inpaint_mask.unsqueeze(0)
            parse_mask_total = parse_mask_total.numpy()
            parse_mask_total = parse_array * parse_mask_total
            parse_mask_total = torch.from_numpy(parse_mask_total)

            # Generate the warped cloth
            # For sake of performance, the TPS parameters are predicted on a low resolution image

            low_cloth = TF.resize(cloth, (256, 192),
                                  T.InterpolationMode.BILINEAR,
                                  antialias=True)
            low_im_mask = TF.resize(im_mask, (256, 192),
                                    T.InterpolationMode.BILINEAR,
                                    antialias=True)
            low_pose_map = TF.resize(pose_map, (256, 192),
                                    T.InterpolationMode.BILINEAR,
                                    antialias=True)
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            agnostic = agnostic.to(device="cpu")
            low_cloth = low_cloth.to(device="cpu")
            low_grid, theta, rx, ry, cx, cy, rg, cg = self.tps(low_cloth.to(torch.float32), agnostic.to(torch.float32))

            # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
            highres_grid = TF.resize(low_grid.permute(0, 3, 1, 2),
                                                                    size=(512, 384),
                                                                    interpolation=T.InterpolationMode.BILINEAR,
                                                                    antialias=True).permute(0, 2, 3, 1)

            warped_cloth = F.grid_sample(cloth.to(torch.float32), highres_grid.to(torch.float32), padding_mode='border')

            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth.to(torch.float32))
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth.to(self.weight_dtype)

        # if "dense_uv" in self.outputlist:
        #     uv = np.load(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5_uv.npz')))
        #     uv = uv['uv']
        #     uv = torch.from_numpy(uv)
        #     uv = transforms.functional.resize(uv, (self.height, self.width))

        # if "dense_labels" in self.outputlist:
        #     labels = Image.open(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
        #     labels = labels.resize((self.width, self.height), Image.NEAREST)
        #     labels = np.array(labels)

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result
    
    def get_image_keypoints(self, img):
        estimator = BodyPoseEstimator(pretrained=True)
        keypoints = estimator(img)
        return keypoints

    def get_image_parse(self, img):
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

        inputs = processor(images=img, return_tensors="pt")

        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        print(pred_seg.shape)
        return pred_seg
