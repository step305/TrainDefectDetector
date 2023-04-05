import json
import os

import imgaug
import imgaug.augmenters as iaa
import labelme2coco
import numpy as np
from PIL import Image, ImageDraw

from backend import rcnn
from backend import rcnn_config
from backend.rcnn import MaskRCNN
from backend.rcnn import load_image_gt
from backend.rcnn import mold_image
from backend.utils import Dataset
from backend.utils import compute_ap

"""
Based on PixelLib code from Waleed Abdulla
"""


class RCNNTraining:
    def __init__(self):
        self.dataset_test = None
        self.dataset_train = None
        self.work_dir = os.getcwd()
        self.config = rcnn_config.RCNNConfig()
        self.model = None

    def load_model(self, model_path):
        self.model = rcnn.MaskRCNN(mode="training", model_dir=self.work_dir, config=self.config)
        self.model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                                                                   "mrcnn_mask"])

    def load_dataset(self, dataset):
        labelme_folder1 = os.path.abspath(os.path.join(dataset, "train"))
        save_json_path1 = os.path.abspath(os.path.join(dataset, "train.json"))
        labelme2coco.convert(labelme_folder1, save_json_path1)

        self.dataset_train = Data()
        self.dataset_train.load_data(save_json_path1, labelme_folder1)
        self.dataset_train.prepare()

        labelme_folder2 = os.path.abspath(os.path.join(dataset, "test"))
        save_json_path2 = os.path.abspath(os.path.join(dataset, "test.json"))
        labelme2coco.convert(labelme_folder2, save_json_path2)

        self.dataset_test = Data()
        self.dataset_test.load_data(save_json_path2, labelme_folder2)
        self.dataset_test.prepare()

    def train_model(self, num_epochs, path_trained_models, layers="all"):
        augmentation = imgaug.augmenters.Sometimes(0.5, [
            imgaug.augmenters.Fliplr(0.5),
            iaa.Flipud(0.5),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
        ])
        self.model.train(self.dataset_train, self.dataset_test, models=path_trained_models,
                         augmentation=augmentation,
                         epochs=num_epochs, layers=layers)

    def evaluate_model(self, model_path, iou_threshold=0.5):
        self.model = MaskRCNN(mode="inference", model_dir=os.getcwd(), config=self.config)
        assert os.path.isfile(model_path) or os.path.isdir(model_path)
        if os.path.isfile(model_path):
            model_files = [model_path]
        elif os.path.isdir(model_path):
            model_files = sorted([os.path.join(model_path, file_name) for file_name in os.listdir(model_path)])
        else:
            raise RuntimeError("wrong model_files")
        for model_file in model_files:
            if str(model_file).endswith(".h5"):
                self.model.load_weights(model_file, by_name=True)
            APs = []
            for image_id in self.dataset_test.image_ids:
                image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self.dataset_test, self.config,
                                                                                 image_id)
                scaled_image = mold_image(image, self.config)
                sample = np.expand_dims(scaled_image, 0)
                yhat = self.model.detect(sample, verbose=0)
                r = yhat[0]
                AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"],
                                         r['masks'],
                                         iou_threshold=iou_threshold)
                APs.append(AP)
            mAP = np.mean(APs)
            print(model_file, "evaluation using iou_threshold", iou_threshold, "is", f"{mAP:01f}", '\n')


class Data(Dataset):
    def load_data(self, annotation_json, images_path):
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        source_name = "coco_like_dataset"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return
            self.add_class(source_name, class_id, class_name)

        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                    continue

                image_path = os.path.abspath(os.path.join(images_path, image_file_name))
                image_annotations = annotations[image_id]

                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids
