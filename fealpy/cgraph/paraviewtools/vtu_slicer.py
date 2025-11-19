"""VTU clipping node - retain half a VTU dataset via planar clip."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Mapping

from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ["VTUSlicer"]


class VTUSlicer(CNodeType):
    r"""Clip a VTU dataset with a plane, retaining half of the volume.

    Inputs:
        dataset (tensor): 上游节点输出的 VTK 数据集。
        enable_slice (menu): 是否执行裁剪（否/是）。
        plane_normal (string): 平面法向量 "x,y,z"。
        plane_ratio (float): 沿法向的裁剪比例（0-100，百分比）。

    Outputs:
        sliced_dataset (tensor): 裁剪后的数据集（或原数据集）。
        slice_info (string): 裁剪操作的摘要信息。
    """

    TITLE: str = "VTU裁剪"
    PATH: str = "后处理.ParaView"
    DESC: str = "对 VTU 数据进行平面裁剪，保留法向一致的一半体积。"
    INPUT_SLOTS = [
        PortConf("dataset", DataType.TENSOR, ttype=1, desc="来自 VTUReader 的数据集", title="数据集"),
           PortConf("enable_slice", DataType.MENU, 0, desc="是否启用裁剪", title="启用裁剪", 
                  default="否", items=["否", "是"]),
        PortConf("plane_normal", DataType.STRING, 0, desc="平面法向量，格式：x,y,z", 
                 title="法向量", default="0,0,1"),
        PortConf("plane_ratio", DataType.FLOAT, 0, desc="沿法向裁剪比例(0-100)", 
             title="裁剪比例(%)", default=50.0, min_val=0, max_val=100),
    ]
    OUTPUT_SLOTS = [
        PortConf("sliced_dataset", DataType.TENSOR, desc="切片后的数据集", title="切片数据集"),
        PortConf("slice_info", DataType.STRING, desc="切片操作信息摘要", title="切片信息"),
    ]

    @staticmethod
    def run(
        dataset: object,
        enable_slice: str = "否",
        plane_normal: str = "0,0,1",
        plane_ratio: float = 50.0,
    ) -> tuple[object, str]:
        print(
            "[VTUSlicer] inputs",
            {
                "enable_slice": enable_slice,
                "plane_normal": plane_normal,
                "plane_ratio": plane_ratio,
            },
        )
        style_payload: dict[str, object] | None = None
        dataset_obj = dataset
        active_array: dict[str, object] | None = None

        if isinstance(dataset, Mapping):
            if "dataset" not in dataset:
                raise ValueError(
                    "VTUSlicer received a mapping input but could not locate the 'dataset' entry."
                )
            style_payload = dict(dataset)
            dataset_obj = style_payload.get("dataset")
            active_array = style_payload.get("active_array")

        if dataset_obj is None:
            raise ValueError("VTUSlicer requires a dataset input.")

        if active_array is None:
            attr = getattr(dataset_obj, "__fealpy_active_array", None)
            if isinstance(attr, Mapping):
                active_array = dict(attr)

        # If clipping is disabled, return original dataset
        if enable_slice == "否":
            slice_info = "Clipping disabled: original dataset returned."
            print("[VTUSlicer] disabled, returning original dataset")
            if active_array is not None:
                try:
                    setattr(dataset_obj, "__fealpy_active_array", active_array)
                except AttributeError:
                    pass
            if style_payload is not None:
                style_payload["dataset"] = dataset_obj
                if active_array is not None:
                    style_payload.setdefault("active_array", active_array)
                style_payload["slice_info"] = slice_info
                return style_payload, slice_info
            return dataset_obj, slice_info

        normal_vec = _parse_vector(plane_normal)

        try:
            ratio = float(plane_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid plane_ratio value: {plane_ratio}") from exc

        if not 0.0 <= ratio <= 100.0:
            raise ValueError("plane_ratio must be between 0 and 100")

        position_val = ratio / 100.0

        sliced_dataset = _apply_clip_slice(
            dataset_obj, normal_vec, position_val
        )
        slice_info = (
            f"Clip applied: normal=({plane_normal}), ratio={ratio:.2f}%, side=正侧"
        )
        print("[VTUSlicer] slice_info", slice_info)

        sliced_dataset = _resample_point_arrays(dataset_obj, sliced_dataset)

        if active_array is not None:
            try:
                setattr(sliced_dataset, "__fealpy_active_array", active_array)
            except AttributeError:
                pass

        sliced_dataset = _ensure_point_arrays(sliced_dataset)

        if style_payload is not None:
            style_payload["dataset"] = sliced_dataset
            if active_array is not None:
                style_payload.setdefault("active_array", active_array)
            style_payload["slice_info"] = slice_info
            return style_payload, slice_info

        return sliced_dataset, slice_info
def _apply_clip_slice(
    dataset: object, plane_normal: tuple[float, float, float], plane_position: float
) -> object:
    """Apply a plane-based clip to retain half of the dataset.

    Args:
        dataset: VTK dataset object
        plane_normal: Normal vector components
        plane_position: Position fraction along the normal direction (0..1)

    Returns:
        Clipped VTK dataset
    """
    try:
        import paraview.simple as pvs
        from paraview.servermanager import Fetch
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ParaView is required for clip operations.") from exc

    source_path = getattr(dataset, "__fealpy_vtu_path", None)

    norm_len = sum(component * component for component in plane_normal) ** 0.5
    if norm_len < 1e-10:
        raise ValueError(f"Invalid normal vector (zero length): {plane_normal}")
    normal = tuple(component / norm_len for component in plane_normal)

    invert_flag = 0

    # Determine bounds along clipping axis to convert fraction to coordinate
    if hasattr(dataset, "GetBounds"):
        bounds = dataset.GetBounds()
    else:
        bounds = None

    plane_coord = plane_position
    if bounds is not None and len(bounds) >= 6:
        # Identify dominant axis of the normal to select min/max pair
        axis_index = max(range(3), key=lambda idx: abs(normal[idx]))
        min_val = bounds[axis_index * 2]
        max_val = bounds[axis_index * 2 + 1]
        plane_coord = min_val + plane_position * (max_val - min_val)

        origin = [0.0, 0.0, 0.0]
        origin[axis_index] = plane_coord
    else:
        origin = [normal[i] * plane_coord for i in range(3)]

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vtu = Path(tmpdir) / "temp_clip_dataset.vtu"
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(temp_vtu))
        prepared_dataset = _augment_dataset_with_cell_arrays(dataset)
        writer.SetInputData(prepared_dataset)
        writer.Write()

        pvs._DisableFirstRenderCameraReset()

        reader = pvs.XMLUnstructuredGridReader(FileName=[str(temp_vtu)])
        reader.UpdatePipeline()

        clip = pvs.Clip(Input=reader, ClipType="Plane")
        clip.ClipType.Normal = normal
        clip.ClipType.Origin = origin
        clip.Invert = invert_flag

        clip.UpdatePipeline()

        result = Fetch(clip)
        if source_path is not None:
            try:
                setattr(result, "__fealpy_vtu_path", source_path)
            except AttributeError:
                pass

        pvs.Delete(clip)
        pvs.Delete(reader)

    return _clone_dataset(result, source_path)


def _parse_vector(vector_str: str) -> tuple[float, float, float]:
    """Parse a vector string in 'x,y,z' format.
    
    Args:
        vector_str: Vector as "x,y,z" string
    
    Returns:
        Tuple of (x, y, z) floats
    """
    parts = [p.strip() for p in vector_str.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Vector must be in 'x,y,z' format, got: {vector_str}"
        )
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f"Invalid vector components (must be floats): {vector_str}"
        ) from exc


def _clone_dataset(dataset: object, source_path: str | None = None) -> object:
    if dataset is None or not hasattr(dataset, "NewInstance"):
        return dataset

    clone = dataset.NewInstance()
    clone.DeepCopy(dataset)

    if source_path is None:
        source_path = getattr(dataset, "__fealpy_vtu_path", None)

    if source_path is not None:
        try:
            setattr(clone, "__fealpy_vtu_path", source_path)
        except AttributeError:
            pass

    active_array = getattr(dataset, "__fealpy_active_array", None)
    if active_array is not None:
        try:
            setattr(clone, "__fealpy_active_array", active_array)
        except AttributeError:
            pass

    return clone


def _resample_point_arrays(source_dataset: object, sliced_dataset: object) -> object:
    if source_dataset is None or sliced_dataset is None:
        return sliced_dataset

    if not hasattr(source_dataset, "GetPointData") or not hasattr(sliced_dataset, "GetPointData"):
        return sliced_dataset

    source_point_data = source_dataset.GetPointData()
    if source_point_data is None or source_point_data.GetNumberOfArrays() == 0:
        return sliced_dataset

    try:
        from vtkmodules.vtkFiltersGeneral import vtkProbeFilter
    except (ModuleNotFoundError, ImportError):
        return sliced_dataset

    probe = vtkProbeFilter()
    probe.SetSourceData(source_dataset)
    probe.SetInputData(sliced_dataset)

    if hasattr(probe, "PassPointArraysOn"):
        probe.PassPointArraysOn()
    if hasattr(probe, "PassCellArraysOn"):
        probe.PassCellArraysOn()
    if hasattr(probe, "PassFieldArraysOn"):
        probe.PassFieldArraysOn()

    probe.Update()
    probed_output = probe.GetOutput()
    if probed_output is None or not hasattr(probed_output, "NewInstance"):
        return sliced_dataset

    cloned = probed_output.NewInstance()
    cloned.DeepCopy(probed_output)

    for attr_name in ("__fealpy_vtu_path", "__fealpy_active_array"):
        attr_value = getattr(sliced_dataset, attr_name, None)
        if attr_value is None:
            attr_value = getattr(source_dataset, attr_name, None)
        if attr_value is not None:
            try:
                setattr(cloned, attr_name, attr_value)
            except AttributeError:
                pass

    return cloned


def _augment_dataset_with_cell_arrays(dataset: object) -> object:
    if dataset is None or not hasattr(dataset, "GetPointData"):
        return dataset

    try:
        from vtkmodules.vtkFiltersCore import vtkPointDataToCellData
    except ModuleNotFoundError:
        return dataset

    converter = vtkPointDataToCellData()
    converter.SetInputData(dataset)
    if hasattr(converter, "PassPointDataOn"):
        converter.PassPointDataOn()
    if hasattr(converter, "ProcessAllArraysOn"):
        converter.ProcessAllArraysOn()

    converter.Update()
    converted = converter.GetOutput()
    if converted is None or not hasattr(converted, "NewInstance"):
        return dataset

    clone = converted.NewInstance()
    clone.DeepCopy(converted)

    for attr_name in ("__fealpy_vtu_path", "__fealpy_active_array"):
        attr_value = getattr(dataset, attr_name, None)
        if attr_value is not None:
            try:
                setattr(clone, attr_name, attr_value)
            except AttributeError:
                pass

    return clone


def _ensure_point_arrays(dataset: object) -> object:
    if dataset is None or not hasattr(dataset, "GetCellData"):
        return dataset

    try:
        from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
    except ModuleNotFoundError:
        return dataset

    cell_data = dataset.GetCellData()
    if cell_data is None or cell_data.GetNumberOfArrays() == 0:
        return dataset

    point_data = dataset.GetPointData() if hasattr(dataset, "GetPointData") else None
    missing_arrays: list[str] = []

    for idx in range(cell_data.GetNumberOfArrays()):
        array = cell_data.GetArray(idx)
        name = array.GetName() if array is not None else cell_data.GetArrayName(idx)
        if not name:
            continue
        if point_data is None or point_data.HasArray(name) != 1:
            missing_arrays.append(name)

    if not missing_arrays:
        return dataset

    converter = vtkCellDataToPointData()
    converter.SetInputData(dataset)
    converter.PassCellDataOn()
    converter.Update()

    converted = converter.GetOutput()
    clone = converted.NewInstance()
    clone.DeepCopy(converted)

    source_path = getattr(dataset, "__fealpy_vtu_path", None)
    if source_path is not None:
        try:
            setattr(clone, "__fealpy_vtu_path", source_path)
        except AttributeError:
            pass

    active_array = getattr(dataset, "__fealpy_active_array", None)
    if active_array is not None:
        try:
            setattr(clone, "__fealpy_active_array", active_array)
        except AttributeError:
            pass

    return clone
