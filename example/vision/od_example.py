#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Object Detection Demo for Basketball Detection using FCOS.

This script demonstrates object detection on a series of images using the FCOS model.
It supports custom resource paths for configuration, weights, and datasets.
"""

import argparse
import os
import time
from typing import List

import cv2
from fcos.config import cfg
from fcos.demo import COCODemo


# Global constants
DEFAULT_RESOURCE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)), 
    '..', 
    'fcos_resources'
))
"""Default resource directory containing FCOS configuration, weights and datasets."""


def demo() -> None:
    """
    Main function to run object detection on a series of images.
    
    This function:
    1. Parses command-line arguments for resource paths and configuration
    2. Loads and configures the FCOS model
    3. Processes each image in the specified directory
    4. Saves detection results to the output directory
    
    Command-line arguments allow customization of resource locations and model parameters.
    """
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Demo")
    
    # Resource path arguments
    parser.add_argument(
        "--resource-root",
        default=DEFAULT_RESOURCE_ROOT,
        metavar="DIR",
        help="Root directory containing FCOS resources (default: %(default)s)"
    )   
    parser.add_argument(
        "--config-file",
        default=os.path.join(DEFAULT_RESOURCE_ROOT, "configs", "Basketball.yaml"),
        metavar="FILE",
        help="Path to model configuration file (default: %(default)s)"
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(DEFAULT_RESOURCE_ROOT, "output_Basketball", "model_final.pth"),
        metavar="FILE",
        help="Path to model weights file (default: %(default)s)"
    )
    parser.add_argument(
        "--images-dir",
        default=os.path.join(DEFAULT_RESOURCE_ROOT, "datasets", "Basketball", "TL_jpg"),
        metavar="DIR",
        help="Directory containing input images (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DEFAULT_RESOURCE_ROOT, "output_Basketball", "output_TL"),
        metavar="DIR",
        help="Directory to save detection results (default: %(default)s)"
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,  
        help="Minimum input image size for the model (default: %(default)d)"
    )
    parser.add_argument(
        "opts",
        help="Command-line options to override configuration values (e.g. `MODEL.XXX 123`)",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Update paths if custom resource root is provided
    if args.resource_root != DEFAULT_RESOURCE_ROOT:
        print(f"[INFO] Using custom resource root: {args.resource_root}")
        args.config_file = os.path.join(args.resource_root, "configs", "Basketball.yaml")
        args.weights = os.path.join(args.resource_root, "output_Basketball", "model_final.pth")
        args.images_dir = os.path.join(args.resource_root, "datasets", "Basketball", "TL_jpg")
        args.output_dir = os.path.join(args.resource_root, "output_Basketball", "output_TL")

    # Load and freeze model configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights
    cfg.freeze()
    print(f"[INFO] Model configuration loaded from: {args.config_file}")

    # Class confidence thresholds (pre-calculated by maximizing per-class F1 scores)
    thresholds_for_classes: List[float] = [
        0.4923645853996277, 0.4928510785102844, 0.5040897727012634,
        0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
        0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
        0.43564629554748535, 0.6089804172515869, 0.666087806224823,
        0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
        0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
        0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
        0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
        0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
        0.523737907409668, 0.47027698159217834, 0.5103300213813782,
        0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
        0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
        0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
        0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
        0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
        0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
        0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
        0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
        0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
        0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
        0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
        0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
        0.5164387822151184, 0.540651798248291, 0.5323763489723206,
        0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
        0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
        0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
        0.4224434792995453, 0.4998510777950287
    ]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {os.path.abspath(args.output_dir)}")

    # Get list of images to process
    image_names: List[str] = [f for f in os.listdir(args.images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] Found {len(image_names)} images in: {args.images_dir}")

    # Initialize object detection demo
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    # Process each image
    for image_name in image_names:
        image_path: str = os.path.join(args.images_dir, image_name)
        image = cv2.imread(image_path)  
        
        # Skip if image couldn't be loaded
        if image is None:
            print(f"[WARNING] Could not load image: {image_path}")
            continue

        # Perform object detection
        start_time: float = time.time()
        result_image = coco_demo.run_on_opencv_image(image)  
        inference_time: float = time.time() - start_time  
        print(f"{image_name} \tInference time: {inference_time:.2f}s")

        # Save results
        output_path: str = os.path.join(args.output_dir, image_name)
        cv2.imwrite(output_path, result_image)  
        print(f"Results saved to: {output_path}")

    print(f"[SUCCESS] All images processed. Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    demo()