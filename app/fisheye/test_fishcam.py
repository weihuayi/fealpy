import cv2
import numpy as np

import ipdb

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
        self.create_blend_mask()
        self.gen_scale_map()
        # 读取刚性 MLS 插值网格
        self.read_mls_grids()

    def fish_to_map(self):
        """
        @brief 
        """
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

        flag = theta < 0
        theta[flag] *= -1
        phi[flag] += np.pi

        flag = theta > np.pi
        theta[flag] = 2*np.pi - theta[flag]
        phi[flag] += np.pi

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

    def create_blend_mask(self):
        """
        创建用于图像混合的遮罩。
        更新成员 `m_blend_post` 和 `m_binary_mask`。
        """
        ws2 = self.ws // 2
        hs2 = self.hs // 2

        # 使用内部遮罩创建环形遮罩
        cir_mask_inv = cv2.bitwise_not(self.inner_cir_mask)
        ring_mask = cv2.bitwise_and(self.cir_mask, cir_mask_inv)

        # 将环形遮罩展开为等矩形投影
        ring_mask_unwarped = cv2.remap(ring_mask, self.map_x, self.map_y, cv2.INTER_LINEAR)

        # 剪切展开后的遮罩
        mask = ring_mask_unwarped[:, ws2 - hs2:ws2 + hs2].astype(np.uint8)

        h, w = mask.shape
        first_zero_col = 120  # 根据 C++ 代码中的固定值
        first_zero_row = 45   # 根据 C++ 代码中的固定值

        # 使用 NumPy 来找到每行第一个零值的位置
        # 将目标区域限制在 [first_zero_col-10, w//2+10]
        cols = np.arange(first_zero_col - 10, w // 2 + 10)
        blend_post = np.zeros(h, dtype=np.int32)

        # 对每一行执行 NumPy 逻辑操作
        for r in range(h):
            if r > h - first_zero_row:
                blend_post[r] = 0
            else:
                row_vals = mask[r, cols]
                zero_indices = np.where(row_vals == 0)[0]
                if zero_indices.size > 0:
                    blend_post[r] = cols[zero_indices[0]] - 15
                else:
                    blend_post[r] = 0

        # 更新类属性
        self.m_blend_post = blend_post.tolist()
        self.m_binary_mask = mask

    def gen_scale_map(self):
        """
        生成用于光衰减补偿的标度图 `m_scale_map`。
        """
        h = self.hs
        w = self.ws
        ws2 = self.ws // 2
        hs2 = self.hs // 2

        # 生成反向的光衰减轮廓 R_pf
        x_coor = np.arange(ws2, dtype=np.float32)
        r_pf = (
            self.p1_ * np.power(x_coor, 5.0) +
            self.p2_ * np.power(x_coor, 4.0) +
            self.p3_ * np.power(x_coor, 3.0) +
            self.p4_ * np.power(x_coor, 2.0) +
            self.p5_ * x_coor +
            self.p6_
        )

        # 元素取倒数
        r_pf = 1.0 / r_pf

        # 创建 IV 象限标度图
        scale_map_quad_4 = np.zeros((hs2, ws2), dtype=np.float32)
        da = r_pf[-1]
        for x in range(ws2):
            for y in range(hs2):
                r = np.floor(np.sqrt(x**2 + y**2))
                if r >= ws2 - 1:
                    scale_map_quad_4[y, x] = da
                else:
                    a = r_pf[int(r)]
                    b = r_pf[min(int(r) + 1, ws2 - 1)]
                    scale_map_quad_4[y, x] = (a + b) / 2.0

        # 生成其他象限的标度图并合并
        scale_map_quad_1 = np.flipud(scale_map_quad_4)
        scale_map_quad_3 = np.fliplr(scale_map_quad_4)
        scale_map_quad_2 = np.fliplr(scale_map_quad_1)

        quad_21 = np.hstack((scale_map_quad_2, scale_map_quad_1))
        quad_34 = np.hstack((scale_map_quad_3, scale_map_quad_4))

        # 将四个象限组合成完整的标度图
        self.m_scale_map = np.vstack((quad_21, quad_34))

    def read_mls_grids(self):
        """
        从文件中读取刚性 MLS 插值网格并更新 `mls_map_x` 和 `mls_map_y`。
        """
        # 打开 MLS 文件
        fs = cv2.FileStorage(self.map_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f"Cannot open map file: {self.map_path}")

        # 读取 `Xd` 和 `Yd` 的值
        mls_map_x = fs.getNode("Xd").mat()
        mls_map_y = fs.getNode("Yd").mat()
        fs.release()

        # 确保读取数据的类型正确
        if mls_map_x is None or mls_map_y is None:
            raise ValueError(f"Missing Xd or Yd data in file: {self.map_path}")

        # 更新类属性
        self.mls_map_x = mls_map_x
        self.mls_map_y = mls_map_y

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


if __name__ == "__main__":
    video_path = '/home/why/data/input_video.mp4'
    mls_map_path = '/home/why/data/grid_xd_yd_3840x1920.yml.gz'
    ipdb.set_trace()
    stitcher = FisheyeStitcher(3840, 1920, 195.0, True, True, mls_map_path)


