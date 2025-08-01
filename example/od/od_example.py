# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import time

from fealpy.od.fcos_core.config import cfg
from fealpy.od.predictor import COCODemo


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Demo")
    parser.add_argument(
        "--config-file",
        default="fealpy/od/FCOS/configs/fcos/Basketball.yaml",  
        metavar="FILE",
        help="模型配置文件路径（.yaml 定义模型、预处理等）",
    )
    parser.add_argument(
        "--weights",
        default="fealpy/od/FCOS/output_Basketball/model_final.pth",  
        metavar="FILE",
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--images-dir",
        default="fealpy/od/FCOS/datasets/Basketball/TL_jpg",  
        metavar="DIR",
        help="待检测图片目录",
    )
    parser.add_argument(
        "--output-dir",
        default="fealpy/od/FCOS/output_Basketball/output_TL",  
        metavar="DIR",
        help="检测结果保存目录",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,  
        help="模型输入最小尺寸（与训练一致以保精度）",
    )
    parser.add_argument(
        "opts",
        help="命令行覆盖配置项（如 `MODEL.XXX 123`）",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # 加载并冻结配置（文件 + 命令行参数）
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights
    cfg.freeze()

    # 类别置信度阈值（通过最大化每类F1分数计算）
    thresholds_for_classes = [
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

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取待检测图片列表
    demo_im_names = os.listdir(args.images_dir)

    # 初始化目标检测演示类
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    # 逐图推理
    for im_name in demo_im_names:
        img_path = os.path.join(args.images_dir, im_name)
        img = cv2.imread(img_path)  
        if img is None:
            continue

        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)  
        inference_time = time.time() - start_time  
        print(f"{im_name} \t推理时间: {inference_time:.2f}s")

        # 保存结果
        output_path = os.path.join(args.output_dir, im_name)
        cv2.imwrite(output_path, composite)  
        print(f"结果已保存至: {output_path}")

    print(f"所有图片处理完成，结果保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()