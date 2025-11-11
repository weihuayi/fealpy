from fealpy.backend import bm
from fealpy.mesher import STPSurfaceMesher


if __name__ == "__main__":
    stp_file = "./data/1.stp"  # 替换为您的 STP 文件路径
    mesher = STPSurfaceMesher(stp_file)
    mesher.init_mesh(min_size=0.05, max_size=5, output_base='./data/云泊1')