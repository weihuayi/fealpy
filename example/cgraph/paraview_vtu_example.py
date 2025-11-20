"""VTU pipeline example without CLI wrapper (reader → styler → screenshot)."""

import json
from pathlib import Path
import shutil

import fealpy.cgraph as cgraph


class _VTUCopyMesh:
    """Lightweight mesh wrapper that copies an existing VTU file."""

    def __init__(self, source: Path):
        self.source = Path(source).expanduser().resolve()
        self.nodedata: dict[str, object] = {}

    def to_vtk(self, fname: str) -> None:
        target = Path(fname)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source, target)

WORLD_GRAPH = cgraph.WORLD_GRAPH

script_dir = Path(__file__).resolve().parent

# Prepare source VTU file that TO_VTK will copy from
source_vtu = Path("C:/Users/thlcz/Downloads/dld_chip_3d.vtu")
if not source_vtu.exists():
    raise FileNotFoundError(f"Sample VTU file missing: {source_vtu}")

export_dir = Path(source_vtu.parent).expanduser().resolve()
exported_vtu_path = export_dir / "test.vtu"
report_png_path = export_dir / "report" / f"{exported_vtu_path.stem}.png"

mesh_stub = _VTUCopyMesh(source_vtu)

# Create nodes: TO_VTK -> VTUReader -> VTUStyler -> VTUScreenshot
exporter = cgraph.create("TO_VTK")
reader = cgraph.create("VTUReader")
styler = cgraph.create("VTUStyler")
screenshot = cgraph.create("VTUScreenshot")

exporter(mesh=mesh_stub, uh=0.0, path=str(export_dir))
# Reader node loads the VTU dataset produced by TO_VTK
reader(vtu_path=exporter().path)

# Apply styling metadata
styler(
    dataset=reader().dataset,
    array_name="uh",
    array_location="POINTS",
    representation="Surface",
    background="#ffffff",
    transparent="否",
    show_scalar_bar="否",
)

# Configure screenshot dimensions
screenshot(
    styled_dataset=styler().styled_dataset,
    image_width=1920,
    image_height=1080,
    output_path=exporter().path,
)

WORLD_GRAPH.output(png_path=screenshot().png_path, vtu_path=exporter().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()

result = WORLD_GRAPH.get()
if not isinstance(result, dict) or "png_path" not in result or "vtu_path" not in result:
    raise RuntimeError(f"Graph execution failed: {result}")

png_path = Path(result["png_path"]).expanduser().resolve()
exported_vtu = Path(result["vtu_path"]).expanduser().resolve()

if not exported_vtu.exists():
    raise FileNotFoundError(f"VTU export failed: {exported_vtu}")

if not png_path.exists():
    raise FileNotFoundError(f"PNG export failed: {png_path}")

if png_path != report_png_path.resolve():
    raise RuntimeError(f"PNG path mismatch: expected {report_png_path}, got {png_path}")

result["png_path"] = str(png_path)
print(json.dumps(result, ensure_ascii=False, indent=2))
