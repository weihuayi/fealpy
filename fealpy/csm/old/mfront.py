import os
import subprocess
import shutil

def compile_mfront_file(file_path):
    """
    @brief 编译 mfront 文件

    TODO:
    1. 测试 mfront 是否已经安装，如果没有安装，提示用户安装，并给出安装过程指导
    2. 检查需要编译的库文件是否已经存在？若存在，问用户是否要重新编译
    3. 考虑一次编译多个 mfront 文件的情形
    """
    # 检查是否安装了 MFront
    if not is_mfront_installed():
        print("未检测到 MFront 的安装，请按照以下指导安装：")
        print_installation_guide()
        return

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"文件 '{file_path}' 不存在")
        return

    # 获取生成的.so文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    so_file = f"lib{base_name}.so"

    # 检查生成的库文件是否已经存在
    if os.path.isfile(so_file):
        print(f"库文件 '{so_file}' 已经存在，无需编译")
        return so_file

    # 构建命令行命令
    command = f"mfront --obuild --interface=generic {file_path}"

    # 运行命令
    try:
        subprocess.check_output(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"编译失败：{e}")
        return

    # 重命名.so文件
    os.rename("src/libBehaviour.so", so_file)

    print(f"编译成功，生成的.so文件名为：{so_file}")

    # 删除include和src文件夹
    shutil.rmtree("include")
    shutil.rmtree("src")

    return so_file

def is_mfront_installed():
    """
    @brief 检查是否安装了 MFront
    """
    try:
        subprocess.check_output("mfront --version", shell=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def print_installation_guide():
    """
    打印安装指南
    """
    with open("mfront_installation_guide.md", "r") as file:
        guide = file.read()
    print(guide)

def compile_mfront_files(file_paths):
    """
    @brief 编译多个 mfront 文件
    ---
    Notes:
    file_paths = ["file1.mfront", "file2.mfront", "file3.mfront"]
    ----
    """
    compiled_files = []

    for file_path in file_paths:
        compiled_file = compile_mfront_file(file_path)
        if compiled_file is not None:
            compiled_files.append(compiled_file)
    return compiled_files


