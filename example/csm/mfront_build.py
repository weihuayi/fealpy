import os
import subprocess
import shutil

def compile_mfront_file(file_path):
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"文件 '{file_path}' 不存在")
        return

    # 构建命令行命令
    command = f"mfront --obuild --interface=generic {file_path}"

    # 运行命令
    try:
        subprocess.check_output(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"编译失败：{e}")
        return

    # 获取生成的.so文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    so_file = f"lib{base_name}.so"

    # 重命名.so文件
    os.rename("src/libBehaviour.so", so_file)

    print(f"编译成功，生成的.so文件名为：{so_file}")

    # 删除include和src文件夹
    shutil.rmtree("include")
    shutil.rmtree("src")

    return so_file

#file = compile_mfront_file('material/saint_venant_kirchhoff.mfront')
file = compile_mfront_file('material/PhaseFieldDisplacementSpectralSplit.mfront')
