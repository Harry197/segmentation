import argparse
import os
import json
import math
import cv2
import numpy as np
from tqdm import tqdm
from shapely import geometry
from scipy.spatial import KDTree

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex

from deeplabv3 import DeepLabv3Plus


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_points_transform(img, pts, is_order_points=True):
    if is_order_points:
        rect = order_points(pts)
    else:
        rect = np.array([pts[0], pts[1], pts[2], pts[3]]).astype(np.float32)

    tl, tr, br, bl = rect

    if abs(tl[0] - tr[0]) < abs(tl[1] - bl[1]):
        tl, tr, br, bl = tr, br, bl, tl
        rect = np.array([tl, tr, br, bl])

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped, rect.astype(np.int32)

def preprocess(image):
    trans = A.Compose( [A.Resize(384, 384),
                        A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
                         , ToTensorV2()
                         ]
                    )
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image /= 255.0
    image = image.astype(np.float32)
    sample = {'image': image}
    sample = trans(**sample)
    image = sample['image']
    image = image.unsqueeze(dim=0)
    return image

def thresholding(probs: np.ndarray, threshold: float = -1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities output by network.

    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        probs = cv2.GaussianBlur(probs, (5, 5), 0)

        thresh_val, bin_img = cv2.threshold(
            probs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.uint8(bin_img / 255)
    else:
        print("threshold >= 0")
        mask = np.uint8(probs > threshold)

    return mask

def find_boxes(boxes_mask: np.ndarray, mode: str = 'min_rectangle', min_area: float = 0.2,
               p_arc_length: float = 0.01, n_max_boxes=math.inf) -> list:
    """
    Finds the coordinates of the box in the binary image `boxes_mask`.

    :param boxes_mask: Binary image: the mask of the box to find. uint8, 2D array
    :param mode: 'min_rectangle' : minimum enclosing rectangle, can be rotated
                 'rectangle' : minimum enclosing rectangle, not rotated
                 'quadrilateral' : minimum polygon approximated by a quadrilateral
    :param min_area: minimum area of the box to be found. A value in percentage of the total area of the image.
    :param p_arc_length: used to compute the epsilon value to approximate the polygon with a quadrilateral.
                         Only used when 'quadrilateral' mode is chosen.
    :param n_max_boxes: maximum number of boxes that can be found (default inf).
                        This will select n_max_boxes with largest area.
    :return: list of length n_max_boxes containing boxes with 4 corners [[x1,y1], ..., [x4,y4]]
    """

    assert len(boxes_mask.shape) == 2, \
        'Input mask must be a 2D array ! Mask is now of shape {}'.format(boxes_mask.shape)

    contours, _ = cv2.findContours(boxes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    found_boxes = list()

    h_img, w_img = boxes_mask.shape[:2]

    def validate_box(box: np.array):
        """

        :param box: array of 4 coordinates with format [[x1,y1], ..., [x4,y4]]
        :return: (box, area)
        """
        polygon = geometry.Polygon([point for point in box])
        if polygon.area > min_area * boxes_mask.size:
            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], boxes_mask.shape[1]),
                            np.minimum(box[:, 1], boxes_mask.shape[0])), axis=1)

            # return box
            return box, polygon.area

    if mode not in ['quadrilateral', 'min_rectangle', 'rectangle']:
        raise NotImplementedError
    if mode == 'quadrilateral':
        for c in contours:
            epsilon = p_arc_length * cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c, epsilon, True)
            # box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))

            # Find extreme points in Convex Hull
            hull_points = cv2.convexHull(cnt, returnPoints=True)
            # points = cnt
            points = hull_points
            if len(points) > 4:
                # Find closes points to corner using nearest neighbors
                tree = KDTree(points[:, 0, :])
                _, ul = tree.query((0, 0))
                _, ur = tree.query((w_img, 0))
                _, dl = tree.query((0, h_img))
                _, dr = tree.query((w_img, h_img))
                box = np.vstack([points[ul, 0, :], points[ur, 0, :],
                                 points[dr, 0, :], points[dl, 0, :]])
            elif len(hull_points) == 4:
                box = hull_points[:, 0, :]
            else:
                continue
            # Todo : test if it looks like a rectangle (2 sides must be more or less parallel)
            # todo : (otherwise we may end with strange quadrilaterals)
            if len(box) != 4:
                mode = 'min_rectangle'
                print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
            else:
                # found_box = validate_box(box)
                found_boxes.append(validate_box(box))
    if mode == 'min_rectangle':
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            found_boxes.append(validate_box(box))
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            found_boxes.append(validate_box(box))
    # sort by area
    found_boxes = [fb for fb in found_boxes if fb is not None]
    found_boxes = sorted(found_boxes, key=lambda x: x[1], reverse=True)
    if n_max_boxes == 1:
        if found_boxes:
            return found_boxes[0][0]
        else:
            return None
    else:
        return [fb[0] for i, fb in enumerate(found_boxes) if i <= n_max_boxes]


def get_transform(mode='train'):
    if mode=='train':
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.HueSaturationValue(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Cutout(num_holes=10, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.5),
                A.Resize(384, 384),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )
    elif mode=="test":
        return 


class MyDataset(Dataset):
    def __init__(self, root_dir, color_dict):
        self.transform = A.Compose([
            A.Resize(384, 384), 
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ), 
            ToTensorV2()
        ])
        self.color_dict = color_dict
        self.images = []
        self.masks = []
        list_file = os.listdir(os.path.join(root_dir, "images"))
        for file_name in list_file:
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                self.images.append(os.path.join(root_dir, "images", file_name))
                self.masks.append(os.path.join(root_dir, "labels", file_name))

    def __getitem__(self, index):
        try:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            index=index-1
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.color_map(mask, self.color_dict)


        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask, self.images[index]
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def color_map(mask, color_dict):
        color_mask = np.zeros([mask.shape[0], mask.shape[1]])
        check = mask[:,:,1] == 255
        # print('check', check.shape)
        color_mask[check] = 1
        # for idx, item in enumerate(color_dict):
        #     check = mask == color_dict[item]
        #     check = np.logical_and(np.logical_and(mask[:,:,0], mask[:,:,1]), mask[:,:,2])
        #     color_mask[check]= idx+1
        
        return np.uint8(color_mask)


def test(args):
    with open(args.classes_path) as json_file:
        color_dict = json.load(json_file)
    num_classes = len(color_dict)
    
    test_set = MyDataset(
        root_dir=args.test_dir,
        color_dict=color_dict
        )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.worker)
    
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    
    if os.path.isdir(args.checkpoint_path):
        file_list = os.listdir(args.checkpoint_path)
        full_list = [os.path.join(args.checkpoint_path, i) for i in file_list]
        model_path = sorted(full_list, key=os.path.getmtime)[-1]
    else:
        model_path = args.checkpoint_path
        
    model = DeepLabv3Plus(models.resnet50(pretrained=True), num_classes=num_classes+1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print('Loaded checkpoint')

    # Set up metrics
    if num_classes == 1:
        metrics = JaccardIndex(task='binary', num_classes=num_classes)
    else:
        metrics = JaccardIndex(task='multiclass', num_classes=num_classes)

    metrics.to(device)

    print("Eval")
    model.eval()

    iou = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels, image_names = data[0].to(device), data[1].to(device), data[2]
            preds = model(images)
            preds = F.interpolate(preds, size=(384,384), mode='bilinear', align_corners=True)
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            score = metrics(preds, labels)
            iou.append(score.item())

            for i, image_name in enumerate(image_names):
                pred = preds[i]
                probs = pred.cpu().numpy()

                mask = thresholding(probs.astype(int))
                polygons = find_boxes(mask, mode='quadrilateral', min_area=0.01)

                img = cv2.imread(image_name)
                h_org, w_org = img.shape[:-1]

                x_scale = w_org / 384    
                y_scale = h_org / 384

                if len(polygons) > 0:
                    for polygon in polygons:
                        new_polygon = []
                        for x, y in polygon:
                            x_new = int(np.round(x * x_scale))
                            y_new = int(np.round(y * y_scale))
                            new_polygon.append([x_new, y_new])

                        pts = np.array(new_polygon, np.int32)
                        img = cv2.polylines(img, [pts], True, (0,255,0), 2)

                cv2.imwrite(os.path.join(args.output_path, image_name.split('/')[-1]), img)

    miou = np.mean(iou)
    print(f'Eval end. mIoU: {miou}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--classes_path', required=True)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints/')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--worker', type=int, default=4)
    parser.add_argument('--output_path', type=str, default='output')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    test(args=args)