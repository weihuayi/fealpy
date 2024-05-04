import cv2
import cv2
import numpy as np

import cv2
import numpy as np

class FisheyeStitcher:
    def __init__(self, width, height, fovd, enb_light_compen, enb_refine_align, map_path):
        self.width = width
        self.height = height
        self.fovd = fovd
        self.enb_light_compen = enb_light_compen
        self.enb_refine_align = enb_refine_align
        self.map_path = map_path

        # 设定源图像的尺寸
        self.ws = width // 2
        self.hs = height

        # 全景图像尺寸
        self.wd = int(self.ws * 360.0 / self.fovd)
        self.hd = self.wd // 2

        # 初始化遮罩和其他属性
        self.init()

    def init(self):
        # 预定义常量
        self.max_fovd = 195.0
        self.p1_ = -7.5625e-17
        self.p2_ = 1.9589e-13
        self.p3_ = -1.8547e-10
        self.p4_ = 6.1997e-08
        self.p5_ = -6.9432e-05
        self.p6_ = 0.9976

        # 初始化其他属性
        self.map_x = None
        self.map_y = None
        self.scale_map = None
        self.cir_mask = None
        self.inner_cir_mask = None

        # 初始化地图和遮罩
        self.fish_to_map()
        self.create_mask()

    def fish_to_eqt(self, x_dest, y_dest, w_rad):
        phi = x_dest / w_rad
        theta = -y_dest / w_rad + np.pi / 2

        if theta < 0:
            theta = -theta
            phi += np.pi
        if theta > np.pi:
            theta = np.pi - (theta - np.pi)
            phi += np.pi

        s = np.sin(theta)
        v0 = s * np.sin(phi)
        v1 = np.cos(theta)
        r = np.sqrt(v0 ** 2 + v1 ** 2)
        theta = w_rad * np.arctan2(r, s * np.cos(phi))

        x_src = theta * v0 / r
        y_src = theta * v1 / r

        return x_src, y_src

    def fish_to_map(self):
        # 创建鱼眼到球面投影的映射表
        w_rad = self.wd / (2.0 * np.pi)
        w2 = self.wd // 2
        h2 = self.hd // 2
        ws2 = self.ws // 2
        hs2 = self.hs // 2

        # 使用数组化计算
        y_indices, x_indices = np.indices((self.hd, self.wd))
        x_d = x_indices - w2
        y_d = y_indices - h2

        # 使用矢量化运算
        phi = x_d / w_rad
        theta = -y_d / w_rad + np.pi / 2

        theta[theta < 0] *= -1
        phi[theta > np.pi] += np.pi
        theta[theta > np.pi] = np.pi - (theta[theta > np.pi] - np.pi)

        s = np.sin(theta)
        v0 = s * np.sin(phi)
        v1 = np.cos(theta)
        r = np.sqrt(v0 ** 2 + v1 ** 2)
        theta = w_rad * np.arctan2(r, s * np.cos(phi))

        x_src = theta * v0 / r + ws2
        y_src = theta * v1 / r + hs2

        self.map_x = x_src.astype(np.float32)
        self.map_y = y_src.astype(np.float32)

    def create_mask(self):
        # 创建圆形遮罩以裁剪图像数据
        ws2 = self.ws // 2
        hs2 = self.hs // 2

        cir_mask = np.zeros((self.hs, self.ws), dtype=np.uint8)
        inner_cir_mask = np.zeros((self.hs, self.ws), dtype=np.uint8)

        wShift = int((self.ws * (self.max_fovd - 183.0) / self.max_fovd) / 2)

        r1 = ws2
        r2 = ws2 - wShift * 2

        cv2.circle(cir_mask, (ws2, hs2), r1, 255, -1)
        cv2.circle(inner_cir_mask, (ws2, hs2), r2, 255, -1)

        self.cir_mask = cir_mask
        self.inner_cir_mask = inner_cir_mask

    @staticmethod
    def extract_frame_from_video(video_path, frame_number=0):
        """
        @brief 从输入视频文件中提取指定帧。

        @param video_path: 输入视频文件的路径
        @param frame_number: 要提取的帧编号（默认第0帧）
        @return: 返回提取的帧图像（如果存在），否则返回 None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")

        # 设置读取指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        # 释放资源
        cap.release()

        if not ret:
            print(f"Unable to extract frame number {frame_number}")
            return None

        return frame

# 示例使用
# stitcher = FisheyeStitcher(3840, 1920, 195.0, True, True, 'path_to_mls_map')

# 示例使用
# stitcher = FisheyeStitcher(1920, 1980, 195.0, True, True, 'path_to_mls_map')
# extracted_frame = FisheyeStitcher.extract_frame_from_video('/home/why/data/input_video.mp4', 10)
# if extracted_frame is not None:
#    cv2.imshow('Extracted Frame', extracted_frame)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def test_fish_to_eqt():
    """
    测试 `fish_to_eqt` 方法是否正确转换单个点坐标。
    """
    stitcher = FisheyeStitcher(3840, 1920, 195.0, True, True, 'path_to_mls_map')
    # 期望的目标坐标
    x_dest, y_dest = 100, 50
    w_rad = stitcher.wd / (2.0 * np.pi)

    # 获取转换后的坐标
    x_src, y_src = stitcher.fish_to_eqt(x_dest, y_dest, w_rad)

    # 打印结果供手动校验
    print(f'fish_to_eqt: x_src = {x_src}, y_src = {y_src}')


def test_fish_to_map():
    """
    测试 `fish_to_map` 方法的映射结果是否合理。
    """
    stitcher = FisheyeStitcher(3840, 1920, 195.0, True, True, 'path_to_mls_map')
    # 生成投影映射表
    stitcher.fish_to_map()

    # 检查映射表的大小和数据类型
    assert stitcher.map_x.shape == (stitcher.hd, stitcher.wd)
    assert stitcher.map_y.shape == (stitcher.hd, stitcher.wd)
    assert stitcher.map_x.dtype == np.float32
    assert stitcher.map_y.dtype == np.float32

    print(f'fish_to_map: map_x and map_y are correctly sized and typed.')


def test_create_mask():
    """
    测试 `create_mask` 方法生成的遮罩是否符合预期。
    """
    stitcher = FisheyeStitcher(3840, 1920, 195.0, True, True, 'path_to_mls_map')
    stitcher.create_mask()

    # 检查遮罩的大小和数据类型
    assert stitcher.cir_mask.shape == (stitcher.hs, stitcher.ws)
    assert stitcher.inner_cir_mask.shape == (stitcher.hs, stitcher.ws)
    assert stitcher.cir_mask.dtype == np.uint8
    assert stitcher.inner_cir_mask.dtype == np.uint8

    print(f'create_mask: cir_mask and inner_cir_mask are correctly sized and typed.')


# 运行测试
test_fish_to_eqt()
test_fish_to_map()
test_create_mask()