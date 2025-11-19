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
slicer = cgraph.create("VTUSlicer")
styler = cgraph.create("VTUStyler")
screenshot = cgraph.create("VTUScreenshot")

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

# Configure screenshot with camera rotation
screenshot(
    styled_dataset=styler().styled_dataset,
    image_width=1920,
    image_height=1080,
    output_path=str(export_dir),
    camera_rotation=180,
    camera_axis="0,1,0",
)

WORLD_GRAPH.output(png_dir=screenshot().png_path, vtu_path=str(source_vtu))
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()

result = WORLD_GRAPH.get()
if not isinstance(result, dict) or "png_dir" not in result:
    raise RuntimeError(f"Graph execution failed: {result}")

# Get the directory where PNG files were generated
png_dir = Path(result["png_dir"]).expanduser().resolve()
if not png_dir.exists():
    raise FileNotFoundError(f"Output directory not found: {png_dir}")

# List all generated PNG files in the directory
png_files = sorted(png_dir.glob("*.png"))

print(f"\n生成了 {len(png_files)} 张截图:")
for idx, png_path in enumerate(png_files, 1):
    print(f"  {idx}. {png_path.name}")

result["png_dir"] = str(png_dir)
result["png_files"] = [str(p) for p in png_files]
print("\n" + json.dumps(result, ensure_ascii=False, indent=2))
