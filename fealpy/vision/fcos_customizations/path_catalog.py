# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths for dataset and model configuration."""

# 1. Python built-in imports
import os


class DatasetCatalog:
    """
    Centralized catalog for dataset paths and configurations.
    
    Provides a unified interface for accessing dataset metadata and paths.
    Supports both COCO and PascalVOC datasets through a factory pattern.
    
    Attributes:
        DATA_DIR (str): Base directory for all datasets.
        DATASETS (dict): Dictionary mapping dataset names to their configurations.
    """
    
    DATA_DIR = "fealpy/vision/fcos_resources/datasets"
    DATASETS = {
        "coco_Basketball_train": {
            "img_dir": "Basketball/images/train",
            "ann_file": "Basketball/annotations/annotations_train.json"
        },
        "coco_Basketball_val": {
            "img_dir": "Basketball/images/valid",
            "ann_file": "Basketball/annotations/annotations_valid.json"
        },
        "coco_Basketball_test": {
            "img_dir": "Basketball/images/test",
            "ann_file": "Basketball/annotations/annotations_test.json"
        }
    }

    @staticmethod
    def get(name: str) -> dict:
        """
        Retrieves configuration for a specific dataset.
        
        Parameters:
            name (str): Name of the dataset to retrieve.
            
        Returns:
            dict: Dictionary containing dataset configuration with:
                - factory: Dataset class name
                - args: Arguments for dataset initialization
                
        Raises:
            RuntimeError: If the requested dataset is not available.
        """
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = {
                "root": os.path.join(data_dir, attrs["img_dir"]),
                "ann_file": os.path.join(data_dir, attrs["ann_file"]),
            }
            return {
                "factory": "CocoDataset",
                "args": args,
            }
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = {
                "data_dir": os.path.join(data_dir, attrs["data_dir"]),
                "split": attrs["split"],
            }
            return {
                "factory": "PascalVOCDataset",
                "args": args,
            }
        raise RuntimeError(f"Dataset not available: {name}")


class ModelCatalog:
    """
    Centralized catalog for model paths and configurations.
    
    Provides URLs and configurations for pretrained models from Detectron and ImageNet.
    
    Attributes:
        S3_C2_DETECTRON_URL (str): Base URL for Detectron models.
        C2_IMAGENET_MODELS (dict): Mapping of model names to their ImageNet paths.
        C2_DETECTRON_SUFFIX (str): Suffix pattern for Detectron model URLs.
        C2_DETECTRON_MODELS (dict): Mapping of model identifiers to their signatures.
    """
    
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d": "ImageNetPretrained/20171220/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = (
        "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/"
        "generalized_rcnn/model_final.pkl"
    )
    
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # Keypoints models
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name: str) -> str:
        """
        Retrieves URL for a specific model.
        
        Parameters:
            name (str): Name of the model to retrieve.
            
        Returns:
            str: URL to the model file.
            
        Raises:
            RuntimeError: If the requested model is not available.
        """
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError(f"Model not present in the catalog: {name}")

    @staticmethod
    def get_c2_imagenet_pretrained(name: str) -> str:
        """
        Constructs URL for ImageNet pretrained models.
        
        Parameters:
            name (str): Full name of the ImageNet pretrained model.
            
        Returns:
            str: Complete URL to the model file.
        """
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        model_path = ModelCatalog.C2_IMAGENET_MODELS[name]
        return "/".join([prefix, model_path])

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name: str) -> str:
        """
        Constructs URL for Detectron 12_2017 baseline models.
        
        Parameters:
            name (str): Full name of the Detectron model.
            
        Returns:
            str: Complete URL to the model file.
        """
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(
            dataset_tag, dataset_tag
        )
        name = name[len("Caffe2Detectron/COCO/"):]
        model_id, model_name = name.split("/")
        model_name = f"{model_name}.yaml"
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = f"{model_name}.{signature}"
        
        return "/".join([
            prefix, 
            model_id, 
            "12_2017_baselines", 
            unique_name, 
            suffix
        ])