"""VTU slicing node - extract 2D slices or apply threshold filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ["VTUSlicer"]


class VTUSlicer(CNodeType):
    r"""Extract a 2D slice or apply threshold filtering to a VTU dataset.

    Inputs:
        dataset (tensor): VTK dataset from VTUReader.
        enable_slice (menu): Whether to apply slicing (否/是).
        slice_type (menu): Type of slice operation (平面/阈值).
        plane_normal (string): Normal vector of the slicing plane "x,y,z" format.
        plane_position (string): Position along normal axis (scalar value).
        array_name (string): Array name for threshold filtering.
        array_range (string): Range for threshold filtering "min,max" format.

    Outputs:
        sliced_dataset (tensor): VTK dataset after slicing (or original if slicing disabled).
        slice_info (string): Information summary about the slice operation.
    """

    TITLE: str = "VTU切片"
    PATH: str = "后处理.ParaView"
    DESC: str = "对 VTU 数据进行平面切片或阈值过滤提取，生成 2D 切片或子集。"
    INPUT_SLOTS = [
        PortConf("dataset", DataType.TENSOR, ttype=1, desc="来自 VTUReader 的数据集", title="数据集"),
        PortConf("enable_slice", DataType.MENU, 0, desc="是否启用切片", title="启用切片", 
                 default="否", items=["否", "是"]),
        PortConf("slice_type", DataType.MENU, 0, desc="切片类型", title="切片类型",
                 default="平面", items=["平面", "阈值"]),
        PortConf("plane_normal", DataType.STRING, 0, desc="平面法向量，格式：x,y,z", 
                 title="法向量", default="0,0,1"),
        PortConf("plane_position", DataType.STRING, 0, desc="沿法向的位置偏移", 
                 title="平面位置", default="0"),
        PortConf("threshold_array", DataType.STRING, 0, desc="用于阈值过滤的数组名称", 
                 title="阈值数组", default="uh"),
        PortConf("threshold_range", DataType.STRING, 0, desc="阈值范围，格式：min,max", 
                 title="阈值范围", default="0,1"),
    ]
    OUTPUT_SLOTS = [
        PortConf("sliced_dataset", DataType.TENSOR, desc="切片后的数据集", title="切片数据集"),
        PortConf("slice_info", DataType.STRING, desc="切片操作信息摘要", title="切片信息"),
    ]

    @staticmethod
    def run(
        dataset: object,
        enable_slice: str = "否",
        slice_type: str = "平面",
        plane_normal: str = "0,0,1",
        plane_position: str = "0",
        threshold_array: str = "uh",
        threshold_range: str = "0,1",
    ) -> tuple[object, str]:
        try:
            import paraview.simple as pvs
            from paraview.servermanager import Fetch
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("ParaView is required for VTUSlicer.") from exc

        # If slicing is disabled, return original dataset
        if enable_slice == "否":
            return dataset, "Slicing disabled: original dataset returned."

        slice_info = ""

        if slice_type == "平面":
            # Plane slicing
            sliced_dataset = _apply_plane_slice(
                dataset, plane_normal, plane_position
            )
            slice_info = (
                f"Plane slice applied: normal=({plane_normal}), "
                f"position={plane_position}"
            )
        elif slice_type == "阈值":
            # Threshold filtering
            sliced_dataset = _apply_threshold_slice(
                dataset, threshold_array, threshold_range
            )
            slice_info = (
                f"Threshold slice applied: array={threshold_array}, "
                f"range={threshold_range}"
            )
        else:
            raise ValueError(f"Unknown slice_type: {slice_type}")

        return sliced_dataset, slice_info


def _apply_plane_slice(
    dataset: object, plane_normal: str, plane_position: str
) -> object:
    """Apply a plane slice to the dataset.
    
    Args:
        dataset: VTK dataset object
        plane_normal: Normal vector as "x,y,z" string
        plane_position: Position along normal as scalar string
    
    Returns:
        Sliced VTK dataset
    """
    try:
        import paraview.simple as pvs
        from paraview.servermanager import Fetch
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ParaView is required for plane slicing.") from exc

    # Parse inputs
    normal = _parse_vector(plane_normal)
    position = float(plane_position)

    # Normalize the normal vector
    norm_len = sum(x**2 for x in normal) ** 0.5
    if norm_len < 1e-10:
        raise ValueError(f"Invalid normal vector (zero length): {plane_normal}")
    normal = tuple(x / norm_len for x in normal)

    # Write dataset to temporary VTU file
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vtu = Path(tmpdir) / "temp_dataset.vtu"
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(temp_vtu))
        writer.SetInputData(dataset)
        writer.Write()

        pvs._DisableFirstRenderCameraReset()

        # Read back via ParaView
        reader = pvs.XMLUnstructuredGridReader(FileName=[str(temp_vtu)])
        reader.UpdatePipeline()

        # Apply plane slice
        sliced = pvs.Slice(
            Input=reader,
            SliceOffsetValues=[position],
            SliceType="Plane",
        )

        # Set plane normal
        sliced.SliceType.Normal = normal

        sliced.UpdatePipeline()

        # Fetch the result as VTK object
        result = Fetch(sliced)

        # Cleanup
        pvs.Delete(sliced)
        pvs.Delete(reader)

    return result


def _apply_threshold_slice(
    dataset: object, array_name: str, threshold_range: str
) -> object:
    """Apply threshold filtering to the dataset.
    
    Args:
        dataset: VTK dataset object
        array_name: Name of the array to threshold on
        threshold_range: Range as "min,max" string
    
    Returns:
        Filtered VTK dataset
    """
    try:
        import paraview.simple as pvs
        from paraview.servermanager import Fetch
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ParaView is required for threshold slicing.") from exc

    # Parse threshold range
    parts = [p.strip() for p in threshold_range.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"threshold_range must be 'min,max' format, got: {threshold_range}"
        )
    try:
        min_val = float(parts[0])
        max_val = float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid threshold range values: {threshold_range}"
        ) from exc

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    # Write dataset to temporary VTU file
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vtu = Path(tmpdir) / "temp_dataset.vtu"
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(temp_vtu))
        writer.SetInputData(dataset)
        writer.Write()

        pvs._DisableFirstRenderCameraReset()

        # Read back via ParaView
        reader = pvs.XMLUnstructuredGridReader(FileName=[str(temp_vtu)])
        reader.UpdatePipeline()

        # Apply threshold
        thresholded = pvs.Threshold(
            Input=reader,
            Scalars=[array_name],
            ThresholdRange=[min_val, max_val],
        )

        thresholded.UpdatePipeline()

        # Fetch the result as VTK object
        result = Fetch(thresholded)

        # Cleanup
        pvs.Delete(thresholded)
        pvs.Delete(reader)

    return result


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
