import os
import sys
from typing import Dict, List, Optional, Union

import torch
import torchvision
from pycocotools.coco import COCO
from torchvision.datasets.coco import CocoDetection

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.keypoint import PersonKeypoints
from fcos_core.structures.segmentation_mask import SegmentationMask


# Global constants
MIN_KEYPOINTS_PER_IMAGE = 10


def _count_visible_keypoints(annotations: List[Dict]) -> int:
    """
    Counts the number of visible keypoints across all annotations.

    Parameters:
        annotations (List[Dict]): A list of annotation dictionaries, each containing keypoint data.

    Returns:
        int: Total count of visible keypoints across all annotations.
    """
    return sum(
        sum(1 for v in ann["keypoints"][2::3] if v > 0) 
        for ann in annotations
    )


def _has_only_empty_bbox(annotations: List[Dict]) -> bool:
    """
    Checks if all bounding boxes in the annotations are effectively empty.

    Parameters:
        annotations (List[Dict]): A list of annotation dictionaries, each containing bounding box data.

    Returns:
        bool: True if all bounding boxes have width or height <= 1, False otherwise.
    """
    return all(
        any(o <= 1 for o in obj["bbox"][2:]) 
        for obj in annotations
    )


def has_valid_annotation(annotations: List[Dict]) -> bool:
    """
    Determines if the provided annotations are valid for training or evaluation.

    Parameters:
        annotations (List[Dict]): A list of annotation dictionaries.

    Returns:
        bool: True if the annotations are valid, False otherwise.
    """
    # Check for empty annotations
    if len(annotations) == 0:
        return False
    
    # Check for annotations with effectively empty bounding boxes
    if _has_only_empty_bbox(annotations):
        return False
    
    # For keypoint detection tasks, require minimum keypoints
    if "keypoints" not in annotations[0]:
        return True
    
    # Validate keypoint annotations
    if _count_visible_keypoints(annotations) >= MIN_KEYPOINTS_PER_IMAGE:
        return True
    
    return False


class CocoDataset(CocoDetection):
    """
    Custom COCO dataset implementation with enhanced annotation handling.

    This class extends torchvision's CocoDetection to provide:
    1. Filtering of images without valid annotations
    2. Robust handling of segmentation masks
    3. Support for keypoint annotations
    4. Conversion between COCO category IDs and contiguous indices

    Parameters:
        ann_file (str): Path to the COCO annotation file.
        root (str): Root directory where images are stored.
        remove_images_without_annotations (bool): Whether to filter images without valid annotations.
        transforms (callable, optional): Optional transform function to apply to images and targets.
    """
    
    def __init__(
        self, 
        ann_file: str, 
        root: str, 
        remove_images_without_annotations: bool, 
        transforms: Optional[callable] = None
    ):
        super().__init__(root, ann_file)
        
        # Sort indices for reproducibility
        self.ids = sorted(self.ids)
        
        # Filter images without valid annotations if requested
        if remove_images_without_annotations:
            valid_ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                annotations = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(annotations):
                    valid_ids.append(img_id)
            self.ids = valid_ids
        
        # Create mapping between COCO category IDs and contiguous indices
        category_ids = self.coco.getCatIds()
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(category_ids)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        
        # Create image ID to index mapping
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an image and its associated annotations.

        Parameters:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (PIL.Image or Tensor): The loaded image
                - target (BoxList): Annotation data as a BoxList
                - index (int): The original index of the image
        """
        # Load image and annotations
        image, annotations = super().__getitem__(index)
        
        # Filter out crowd annotations
        annotations = [obj for obj in annotations if obj["iscrowd"] == 0]
        
        # Process bounding boxes
        boxes = [obj["bbox"] for obj in annotations]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")
        
        # Process category labels
        category_ids = [obj["category_id"] for obj in annotations]
        contiguous_ids = [
            self.json_category_id_to_contiguous_id[cid] 
            for cid in category_ids
        ]
        labels = torch.tensor(contiguous_ids)
        target.add_field("labels", labels)
        
        # Process segmentation masks with fallback for missing annotations
        if any(len(obj.get("segmentation", [])) > 0 for obj in annotations):
            masks = [obj["segmentation"] for obj in annotations]
            masks = SegmentationMask(masks, image.size, mode='poly')
        else:
            # Create dummy masks when no valid segmentation data exists
            dummy_mask = torch.zeros(
                (len(annotations), *image.size[::-1]), 
                dtype=torch.uint8
            )
            masks = SegmentationMask(dummy_mask, image.size, mode='mask')
        target.add_field("masks", masks)
        
        # Process keypoints if available
        if annotations and "keypoints" in annotations[0]:
            keypoints = [obj["keypoints"] for obj in annotations]
            keypoints = PersonKeypoints(keypoints, image.size)
            target.add_field("keypoints", keypoints)
        
        # Clip boxes to image boundaries and remove empty boxes
        target = target.clip_to_image(remove_empty=True)
        
        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target, index

    def get_img_info(self, index: int) -> Dict:
        """
        Retrieves metadata information for a specific image.

        Parameters:
            index (int): Index of the image.

        Returns:
            Dict: Dictionary containing image metadata (id, width, height, etc.)
        """
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]