"""VTU pipeline example without CLI wrapper (reader → slicer → styler → screenshot)."""

import json
from pathlib import Path

import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "data"
source_vtu = (data_dir / "dld_chip_3d.vtu").resolve()
if not source_vtu.is_file():
    raise FileNotFoundError(f"Sample VTU missing: {source_vtu}")
print(f"Using VTU source: {source_vtu}")

export_dir = (script_dir / "output").resolve()
report_png_path = export_dir / "report" / f"{source_vtu.stem}.png"

# Create nodes: VTUReader -> VTUSlicer -> VTUStyler -> VTUScreenshot
reader = cgraph.create("VTUReader")

# uh 渲染节点
slicer = cgraph.create("VTUSlicer")
styler = cgraph.create("VTUStyler")
screenshot = cgraph.create("VTUScreenshot")

# ph 渲染节点
slicer_ph = cgraph.create("VTUSlicer")
styler_ph = cgraph.create("VTUStyler")
screenshot_ph = cgraph.create("VTUScreenshot")

# Reader node loads the VTU dataset
reader(vtu_path=str(source_vtu))

# Compute z-axis midpoint for slicing
try:
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("VTU slicing requires VTK/ParaView modules.") from exc

vtk_reader = vtkXMLUnstructuredGridReader()
vtk_reader.SetFileName(str(source_vtu))
vtk_reader.Update()
vtk_dataset = vtk_reader.GetOutput()
if vtk_dataset is None:
    raise RuntimeError("Failed to load VTU dataset for slicing bounds.")

# Apply slicing at z midpoint
slicer(
    dataset=reader().dataset,
    enable_slice="是",
    plane_normal="0,0,1",
    plane_ratio=50.0,
)

# ph 切片（参数与uh一致）
slicer_ph(
    dataset=reader().dataset,
    enable_slice="是",
    plane_normal="0,0,1",
    plane_ratio=50.0,
)

# Apply styling metadata to the sliced dataset
styler(
    dataset=slicer().sliced_dataset,
    array_name="uh",
    array_location="POINTS",
    representation="Surface",
    background="#1e1e1e",
    transparent="否",
    show_scalar_bar="是",
)

# ph 样式（仅 array_name 改为 ph）
styler_ph(
    dataset=slicer_ph().sliced_dataset,
    array_name="ph",
    array_location="POINTS",
    representation="Surface",
    background="#1e1e1e",
    transparent="否",
    show_scalar_bar="是",
)


# 规范图片命名（去掉分量数字0）
uh_png_name = f"{source_vtu.stem}_uh.png"
ph_png_name = f"{source_vtu.stem}_ph.png"




# uh 截图（只保留命名规范的调用）
screenshot(
    styled_dataset=styler().styled_dataset,
    image_width=1920,
    image_height=1080,
    output_path=str(export_dir / uh_png_name),
    camera_rotation=180,
    camera_axis="0,1,0",
)


# ph 截图（只保留命名规范的调用）
screenshot_ph(
    styled_dataset=styler_ph().styled_dataset,
    image_width=1920,
    image_height=1080,
    output_path=str(export_dir / ph_png_name),
    camera_rotation=180,
    camera_axis="0,1,0",
)

WORLD_GRAPH.output(
    png_dir=screenshot().png_path,
    png_dir_ph=screenshot_ph().png_path,
    vtu_path=str(source_vtu)
)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()



result = WORLD_GRAPH.get()
if not isinstance(result, dict) or "png_dir" not in result:
    raise RuntimeError(f"Graph execution failed: {result}")

uh_png_path = str(export_dir / uh_png_name)
ph_png_path = str(export_dir / ph_png_name)
print(f"\n已生成图片: {uh_png_path}, {ph_png_path}")
