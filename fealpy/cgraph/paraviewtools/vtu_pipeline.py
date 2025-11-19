"""ParaView-based VTU processing pipeline nodes - decoupled design."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ["VTUReader", "VTUStyler", "VTUScreenshot", "TO_VTK"]


class VTUReader(CNodeType):
    r"""Read a VTU file and return the dataset.

    Inputs:
        vtu_path (string): Path to the VTU file.

    Outputs:
        dataset (object): VTK UnstructuredGrid dataset.
    """

    TITLE: str = "VTU读取"
    PATH: str = "后处理.ParaView"
    DESC: str = "读取 VTU 文件并返回数据集。"
    INPUT_SLOTS = [
        PortConf("vtu_path", DataType.NONE, 0, desc="VTU 文件路径", title="文件路径", default=""),
    ]
    OUTPUT_SLOTS = [
        PortConf("dataset", DataType.TENSOR, desc="VTK 数据集", title="数据集"),
    ]

    @staticmethod
    def run(vtu_path: str):
        print(f"[VTUReader] input vtu_path={vtu_path!r}")
        try:
            from paraview.servermanager import Fetch
            import paraview.simple as pvs
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ParaView is required to run VTUReader. Make sure pvpython is available."
            ) from exc

        # 临时硬编码前端联调默认路径，避免空路径导致找不到文件
        if not vtu_path:
            vtu_path = "/app/fealpy-service/dld_chip_3d.vtu"

        source_path = Path(vtu_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"VTU file not found: {source_path}")

        pvs._DisableFirstRenderCameraReset()
        reader = pvs.XMLUnstructuredGridReader(FileName=[str(source_path)])
        reader.UpdatePipeline()
        dataset = Fetch(reader)
        print(f"[VTUReader] loaded dataset from {source_path}")
        try:
            setattr(dataset, "__fealpy_vtu_path", str(source_path))
        except AttributeError:
            pass
        pvs.Delete(reader)

        return dataset


class VTUStyler(CNodeType):
    r"""Apply styling (representation, colors, background) to a VTU dataset.

    Inputs:
        dataset (tensor): VTK dataset object from VTUReader.
        array_name (string): Name of the data array for colouring.
        array_location (menu): Data array location (POINTS or CELLS).
        representation (menu): Geometric representation.
        background (string): Background colour (#RRGGBB or name).
        transparent (menu): Whether background is transparent.
        show_scalar_bar (menu): Show scalar bar.

    Outputs:
        styled_dataset (tensor): Dataset with applied styling metadata.
        coloring_info (string): Info about applied coloring.
    """

    TITLE: str = "VTU样式化"
    PATH: str = "后处理.ParaView"
    DESC: str = "对 VTU 数据应用样式设置（颜色、表现形式、背景等）。"
    INPUT_SLOTS = [
        PortConf("dataset", DataType.TENSOR, ttype=1, desc="来自 VTUReader 的数据集", title="数据集"),
        PortConf("array_name", DataType.STRING, 0, desc="用于着色的数据场名称", title="数据场", default="uh"),
        PortConf("array_location", DataType.MENU, 0, desc="数据场所属位置", title="数据类型", default="POINTS", items=["POINTS", "CELLS"]),
        PortConf("representation", DataType.MENU, 0, desc="几何显示方式", title="显示模式", default="Surface", items=["Surface", "Surface With Edges", "Wireframe", "Points"]),
        PortConf("background", DataType.STRING, 0, desc="背景颜色", title="背景颜色", default="#ffffff"),
        PortConf("transparent", DataType.MENU, 0, desc="是否透明背景", title="透明背景", default="否", items=["否", "是"]),
        PortConf("show_scalar_bar", DataType.MENU, 0, desc="是否显示图例标尺", title="显示图例", default="否", items=["否", "是"]),
    ]
    OUTPUT_SLOTS = [
        PortConf("styled_dataset", DataType.TENSOR, desc="应用了样式的数据集（含元数据）", title="样式数据集"),
        PortConf("coloring_info", DataType.STRING, desc="着色信息摘要", title="着色信息"),
    ]

    @staticmethod
    def run(
        dataset: object,
        array_name: str = "uh",
        array_location: str = "POINTS",
        representation: str = "Surface",
        background: str | Sequence[float] = "#ffffff",
        transparent: str = "否",
        show_scalar_bar: str = "否",
    ) -> tuple[dict, str]:
        print(
            "[VTUStyler] inputs",
            {
                "array_name": array_name,
                "array_location": array_location,
                "representation": representation,
                "background": background,
                "transparent": transparent,
                "show_scalar_bar": show_scalar_bar,
            },
        )
        try:
            import paraview.simple as pvs
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("ParaView is required for VTUStyler.") from exc

        # Validate inputs
        location = (array_location or "POINTS").upper()
        if location not in {"POINTS", "CELLS"}:
            raise ValueError("array_location must be 'POINTS' or 'CELLS'.")

        valid_repr = {"Surface", "Surface With Edges", "Wireframe", "Points"}
        if representation not in valid_repr:
            raise ValueError(f"representation must be one of {sorted(valid_repr)}")

        _parse_background(background)  # Validate color format

        transparent_bool = True if transparent == "是" else False
        show_scalar_bar_bool = True if show_scalar_bar == "是" else False

        active_array = {
            "name": array_name,
            "location": location,
        }

        dataset = _ensure_point_array(dataset, array_name, location)

        # Store styling metadata in a dict for later use
        styled_dataset_dict = {
            "dataset": dataset,
            "array_name": array_name,
            "array_location": location,
            "representation": representation,
            "background": background,
            "transparent": transparent_bool,
            "show_scalar_bar": show_scalar_bar_bool,
            "vtu_path": getattr(dataset, "__fealpy_vtu_path", None),
            "active_array": active_array,
        }

        try:
            setattr(dataset, "__fealpy_active_array", active_array)
        except AttributeError:
            pass

        coloring_info = (
            f"Array: {array_name} ({location}), "
            f"Representation: {representation}, "
            f"Transparent: {transparent}, Scalar Bar: {show_scalar_bar}"
        )
        print("[VTUStyler] coloring_info", coloring_info)

        return styled_dataset_dict, coloring_info


class VTUScreenshot(CNodeType):
    r"""Render a styled VTU dataset and export to PNG images.
    
    For multi-component arrays (vectors/tensors), generates:
    - Magnitude image
    - Component images (X, Y, Z, etc.)
    
    For scalar arrays, generates a single image.

    Inputs:
        styled_dataset (tensor): VTK dataset from VTUStyler.
        image_width (int): Output image width in pixels.
        image_height (int): Output image height in pixels.
        output_path (string): Optional path hint. When provided, a "report"
            directory will be created under the given path (or its parent when a
            file path is supplied) and used for the PNG output. If empty,
            defaults to the VTU source directory when available, otherwise the
            current working directory.

    Outputs:
        png_path (string): Base directory path where PNG files are generated.
            For multi-component arrays, files are named with suffixes:
            {basename}_magnitude.png, {basename}_X.png, {basename}_Y.png, {basename}_Z.png
            A JSON metadata file ({basename}_info.json) is also generated with details.
    """

    TITLE: str = "VTU截图"
    PATH: str = "后处理.ParaView"
    DESC: str = "渲染样式化的 VTU 数据并导出 PNG 图像。对于多分量数组，自动生成模和各分量图片。"
    INPUT_SLOTS = [
        PortConf("styled_dataset", DataType.TENSOR, ttype=1, desc="来自 VTUStyler 的样式数据集", title="样式数据集"),
        PortConf("image_width", DataType.INT, 0, desc="输出图像宽度", title="图像宽度", default=1920, min_val=1),
        PortConf("image_height", DataType.INT, 0, desc="输出图像高度", title="图像高度", default=1080, min_val=1),
        PortConf("output_path", DataType.STRING, 0, desc="PNG 输出路径 (可选)", title="输出路径", default=""),
        PortConf("camera_rotation", DataType.FLOAT, 0, desc="相机绕指定轴旋转角度 (度)", title="相机旋转角度", default=0.0),
        PortConf("camera_axis", DataType.STRING, 0, desc="相机旋转轴 (格式:x,y,z)", title="相机旋转轴", default="0,0,1"),
    ]
    OUTPUT_SLOTS = [
        PortConf("png_path", DataType.STRING, desc="PNG 文件所在目录路径", title="PNG 目录"),
    ]

    @staticmethod
    def run(
        styled_dataset: dict,
        image_width: int = 1920,
        image_height: int = 1080,
        output_path: str = "",
        camera_rotation: float = 0.0,
        camera_axis: str = "0,0,1",
    ) -> str:
        print(
            "[VTUScreenshot] inputs",
            {
                "output_path": output_path,
                "image_width": image_width,
                "image_height": image_height,
                "camera_rotation": camera_rotation,
                "camera_axis": camera_axis,
            },
        )
        try:
            import paraview.simple as pvs
            from paraview.servermanager import Fetch
            from vtkmodules.vtkIOXML import (
                vtkXMLUnstructuredGridWriter,
                vtkXMLPolyDataWriter,
                vtkXMLImageDataWriter,
                vtkXMLStructuredGridWriter,
                vtkXMLRectilinearGridWriter,
            )
            try:  # vtkXMLGenericDataObjectWriter is absent in older ParaView builds
                from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectWriter  # type: ignore
            except ImportError:  # pragma: no cover - older ParaView
                vtkXMLGenericDataObjectWriter = None  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("ParaView is required for VTUScreenshot.") from exc

        # Extract dataset and styling info from the dict
        vtu_path_hint: str | None = None

        if isinstance(styled_dataset, dict):
            dataset = styled_dataset.get("dataset")
            array_name = styled_dataset.get("array_name", "uh")
            array_location = styled_dataset.get("array_location", "POINTS")
            representation = styled_dataset.get("representation", "Surface")
            background = styled_dataset.get("background", "#ffffff")
            show_scalar_bar = styled_dataset.get("show_scalar_bar", False)
            vtu_path_hint = styled_dataset.get("vtu_path")
        else:
            # Fallback for raw VTK object (backward compatibility)
            dataset = styled_dataset
            array_name = "uh"
            array_location = "POINTS"
            representation = "Surface"
            background = "#ffffff"
            show_scalar_bar = False
            vtu_path_hint = getattr(dataset, "__fealpy_vtu_path", None)

        if not vtu_path_hint and dataset is not None:
            vtu_path_hint = getattr(dataset, "__fealpy_vtu_path", None)

        width = int(image_width)
        height = int(image_height)
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive integers.")

        if output_path:
            target_base = Path(output_path).expanduser()
            suffix = target_base.suffix.lower()
            custom_name: str | None = None

            if suffix == ".png":
                base_dir = target_base.parent
                custom_name = target_base.name
                stem_hint = target_base.stem
            else:
                base_dir = target_base.parent if suffix else target_base
                stem_hint = target_base.stem if suffix else None

            base_dir = base_dir.expanduser()
            report_dir = (base_dir / "report").expanduser()
            report_dir.mkdir(parents=True, exist_ok=True)

            if not custom_name:
                stem_candidates = [
                    stem_hint,
                    Path(vtu_path_hint).stem if isinstance(vtu_path_hint, str) and vtu_path_hint else None,
                    base_dir.name or None,
                ]
                stem = next((s for s in stem_candidates if s), "vtu_screenshot")
                custom_name = f"{stem}.png"

            target_path = (report_dir / custom_name).expanduser()
        else:
            hinted_vtu = vtu_path_hint or ""
            if isinstance(hinted_vtu, str) and hinted_vtu:
                target_path = Path(hinted_vtu).expanduser().with_suffix(".png")
            else:
                target_path = Path.cwd() / "vtu_screenshot.png"

        target_path = target_path.resolve()
        print(f"[VTUScreenshot] resolved output_path={target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if dataset is None:
            raise ValueError("VTUScreenshot requires a dataset to render.")

        temp_dir = target_path.parent
        temp_suffix = ".temp_pipeline.vtu"
        reader_factory = pvs.XMLUnstructuredGridReader

        if hasattr(dataset, "IsA"):
            if dataset.IsA("vtkUnstructuredGrid"):
                writer = vtkXMLUnstructuredGridWriter()
            elif dataset.IsA("vtkPolyData"):
                writer = vtkXMLPolyDataWriter()
                reader_factory = pvs.XMLPolyDataReader
                temp_suffix = ".temp_pipeline.vtp"
            elif dataset.IsA("vtkImageData"):
                writer = vtkXMLImageDataWriter()
                reader_factory = pvs.XMLImageDataReader
                temp_suffix = ".temp_pipeline.vti"
            elif dataset.IsA("vtkStructuredGrid"):
                writer = vtkXMLStructuredGridWriter()
                reader_factory = pvs.XMLStructuredGridReader
                temp_suffix = ".temp_pipeline.vts"
            elif dataset.IsA("vtkRectilinearGrid"):
                writer = vtkXMLRectilinearGridWriter()
                reader_factory = pvs.XMLRectilinearGridReader
                temp_suffix = ".temp_pipeline.vtr"
            else:
                if vtkXMLGenericDataObjectWriter is not None:
                    writer = vtkXMLGenericDataObjectWriter()
                    reader_factory = getattr(pvs, "XMLGenericDataObjectReader", pvs.XMLUnstructuredGridReader)
                    temp_suffix = ".temp_pipeline.vtm"
                else:
                    writer = vtkXMLUnstructuredGridWriter()
                    reader_factory = pvs.XMLUnstructuredGridReader
                    temp_suffix = ".temp_pipeline.vtu"
        else:
            if vtkXMLGenericDataObjectWriter is not None:
                writer = vtkXMLGenericDataObjectWriter()
                reader_factory = getattr(pvs, "XMLGenericDataObjectReader", pvs.XMLUnstructuredGridReader)
                temp_suffix = ".temp_pipeline.vtm"
            else:
                writer = vtkXMLUnstructuredGridWriter()
                reader_factory = pvs.XMLUnstructuredGridReader
                temp_suffix = ".temp_pipeline.vtu"

        temp_vtk_path = temp_dir / temp_suffix
        writer.SetFileName(str(temp_vtk_path))
        writer.SetInputData(dataset)
        writer.Write()

        pvs._DisableFirstRenderCameraReset()

        # Read the temporary file as a ParaView source
        reader = reader_factory(FileName=[str(temp_vtk_path)])
        reader.UpdatePipeline()

        # Fetch dataset for array validation
        vtk_dataset = Fetch(reader)
        
        # Detect number of components in the array
        num_components = 1
        component_names = []
        if array_name and array_name.strip():
            data_attr = vtk_dataset.GetPointData() if array_location == "POINTS" else vtk_dataset.GetCellData()
            if data_attr and data_attr.HasArray(array_name) == 1:
                array_obj = data_attr.GetArray(array_name)
                if array_obj:
                    num_components = array_obj.GetNumberOfComponents()
                    print(f"[VTUScreenshot] Array '{array_name}' has {num_components} components")
                    
                    # Generate component names
                    if num_components == 1:
                        component_names = ["scalar"]
                    elif num_components == 3:
                        component_names = ["magnitude", "X", "Y", "Z"]
                    elif num_components == 2:
                        component_names = ["magnitude", "X", "Y"]
                    else:
                        component_names = ["magnitude"] + [f"Component_{i}" for i in range(num_components)]
        
        # Generate screenshots for each component
        generated_paths = []
        metadata_list = []
        
        for comp_idx, comp_name in enumerate(component_names):
            # Determine output path for this component
            if len(component_names) > 1:
                # Multi-component: add suffix
                comp_target_path = target_path.parent / f"{target_path.stem}_{comp_name}{target_path.suffix}"
            else:
                # Single component: use original path
                comp_target_path = target_path
            
            print(f"[VTUScreenshot] Rendering component: {comp_name} -> {comp_target_path}")
            
            view = pvs.CreateView("RenderView")
            view.ViewSize = [width, height]
            background_rgb = _parse_background(background)
            view.Background = background_rgb

            pvs.SetActiveSource(reader)
            pvs.SetActiveView(view)

            # Show the dataset in the view
            display = pvs.Show(reader, view)
            display.Representation = representation

            # Apply coloring
            if array_name and array_name.strip():
                data_attr = vtk_dataset.GetPointData() if array_location == "POINTS" else vtk_dataset.GetCellData()
                
                if data_attr and data_attr.HasArray(array_name) == 1:
                    # For magnitude (comp_idx == 0) or scalar, use default magnitude mode
                    if comp_idx == 0:
                        pvs.ColorBy(display, (array_location, array_name))
                    else:
                        # For component X, Y, Z (comp_idx 1, 2, 3), use component mode
                        pvs.ColorBy(display, (array_location, array_name, comp_idx - 1))
                    
                    color_tf = pvs.GetColorTransferFunction(array_name)
                    opacity_tf = pvs.GetOpacityTransferFunction(array_name)
                    
                    if comp_idx == 0:
                        # Magnitude: use full range
                        data_range = _array_range(data_attr, array_name)
                    else:
                        # Component: get component-specific range
                        array_obj = data_attr.GetArray(array_name)
                        if array_obj and comp_idx - 1 < array_obj.GetNumberOfComponents():
                            comp_range = array_obj.GetRange(comp_idx - 1)
                            data_range = (float(comp_range[0]), float(comp_range[1]))
                            if data_range[0] == data_range[1]:
                                eps = max(abs(data_range[0]), 1.0) * 1e-6
                                data_range = (data_range[0] - eps, data_range[1] + eps)
                        else:
                            data_range = _array_range(data_attr, array_name)
                    
                    color_tf.RescaleTransferFunction(*data_range)
                    opacity_tf.RescaleTransferFunction(*data_range)

            display.SetScalarBarVisibility(view, show_scalar_bar)

            pvs.ResetCamera(view)
            
            # Apply camera rotation if specified
            if camera_rotation:
                try:
                    import math

                    axis_tokens = [float(part.strip()) for part in camera_axis.split(",")]
                    if len(axis_tokens) != 3:
                        raise ValueError
                    axis_norm = math.sqrt(sum(component * component for component in axis_tokens))
                    if axis_norm < 1e-12:
                        raise ValueError
                    axis_dir = tuple(component / axis_norm for component in axis_tokens)

                    angle_rad = math.radians(float(camera_rotation))
                    if abs(angle_rad) < 1e-12:
                        raise ValueError

                    camera = getattr(view, "GetActiveCamera", lambda: None)()
                    if camera is not None:
                        position = tuple(camera.GetPosition())
                        focal_point = tuple(camera.GetFocalPoint())
                        view_up = tuple(camera.GetViewUp())

                        print(
                            "[VTUScreenshot] camera before rotation",
                            f"pos={position}",
                            f"focal={focal_point}",
                            f"up={view_up}",
                        )

                        if len(position) == 3 and len(focal_point) == 3 and len(view_up) == 3:
                            offset = tuple(p - f for p, f in zip(position, focal_point))

                            def _rotate_vec(vec: tuple[float, float, float]) -> tuple[float, float, float]:
                                vx, vy, vz = vec
                                ax, ay, az = axis_dir
                                cos_a = math.cos(angle_rad)
                                sin_a = math.sin(angle_rad)
                                dot = ax * vx + ay * vy + az * vz
                                cross_x = ay * vz - az * vy
                                cross_y = az * vx - ax * vz
                                cross_z = ax * vy - ay * vx
                                rx = vx * cos_a + cross_x * sin_a + ax * dot * (1.0 - cos_a)
                                ry = vy * cos_a + cross_y * sin_a + ay * dot * (1.0 - cos_a)
                                rz = vz * cos_a + cross_z * sin_a + az * dot * (1.0 - cos_a)
                                return (rx, ry, rz)

                            rotated_offset = _rotate_vec(offset)
                            rotated_up = _rotate_vec(view_up)

                            new_position = tuple(f + r for f, r in zip(focal_point, rotated_offset))
                            camera.SetPosition(*new_position)
                            camera.SetViewUp(*rotated_up)
                            if hasattr(view, "ResetCameraClippingRange"):
                                view.ResetCameraClippingRange()

                            print(
                                "[VTUScreenshot] camera after rotation",
                                f"pos={new_position}",
                                f"focal={focal_point}",
                                f"up={rotated_up}",
                            )
                except (TypeError, ImportError, ValueError):
                    pass
            
            pvs.Render()

            pvs.SaveScreenshot(
                str(comp_target_path),
                view,
                ImageResolution=[width, height],
            )

            pvs.Delete(view)
            
            generated_paths.append(str(comp_target_path))
            
            # Collect metadata for this component
            # Generate physical description
            if comp_idx == 0 and num_components > 1:
                description = f"{array_name} 的模（magnitude），表示向量场的大小或强度"
                physical_meaning = "向量场的模长，反映物理量的总体强度大小，不包含方向信息"
            elif comp_name == "X":
                description = f"{array_name} 的 X 方向分量"
                physical_meaning = "物理量在 X 轴（水平）方向的投影值，正值表示沿 +X 方向，负值表示沿 -X 方向"
            elif comp_name == "Y":
                description = f"{array_name} 的 Y 方向分量"
                physical_meaning = "物理量在 Y 轴（垂直）方向的投影值，正值表示沿 +Y 方向，负值表示沿 -Y 方向"
            elif comp_name == "Z":
                description = f"{array_name} 的 Z 方向分量"
                physical_meaning = "物理量在 Z 轴（深度）方向的投影值，正值表示沿 +Z 方向，负值表示沿 -Z 方向"
            elif comp_name == "scalar":
                description = f"{array_name} 的标量分布"
                physical_meaning = "标量物理量的空间分布，仅有大小没有方向"
            else:
                description = f"{array_name} 的第 {comp_idx} 个分量"
                physical_meaning = f"物理量的第 {comp_idx} 个独立分量"
            
            component_metadata = {
                "file_name": comp_target_path.name,
                "file_path": str(comp_target_path),
                "component_name": comp_name,
                "component_index": comp_idx,
                "is_magnitude": comp_idx == 0 and num_components > 1,
                "description": description,
                "physical_meaning": physical_meaning,
                "array_name": array_name,
                "array_location": array_location,
                "data_range": list(data_range) if 'data_range' in locals() else None,
                "visualization_settings": {
                    "image_width": width,
                    "image_height": height,
                    "representation": representation,
                    "background": background,
                    "show_scalar_bar": show_scalar_bar,
                    "camera_rotation": camera_rotation,
                    "camera_axis": camera_axis,
                },
            }
            metadata_list.append(component_metadata)
        
        pvs.Delete(reader)
        
        # Clean up temporary file
        if temp_vtk_path.exists():
            temp_vtk_path.unlink()

        # Generate JSON metadata file
        import json
        import datetime
        
        # Determine field type description
        if num_components == 1:
            field_type = "标量场（Scalar Field）"
            field_type_desc = "仅具有大小的物理量，例如温度、压强、密度等"
        elif num_components == 2:
            field_type = "二维向量场（2D Vector Field）"
            field_type_desc = "具有大小和方向的物理量，包含 X 和 Y 两个方向分量"
        elif num_components == 3:
            field_type = "三维向量场（3D Vector Field）"
            field_type_desc = "具有大小和方向的物理量，包含 X、Y 和 Z 三个方向分量，例如速度场、力场、位移场等"
        else:
            field_type = f"{num_components} 分量张量场"
            field_type_desc = f"包含 {num_components} 个独立分量的高阶张量物理量"
        
        json_metadata = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "generator": "FEALPy VTUScreenshot Node",
                "version": "1.0"
            },
            "source_data": {
                "vtu_file": vtu_path_hint,
                "array_name": array_name,
                "array_location": array_location,
                "field_type": field_type,
                "field_type_description": field_type_desc,
                "total_components": num_components,
            },
            "output_info": {
                "output_directory": str(target_path.parent),
                "base_name": target_path.stem,
                "total_images": len(generated_paths),
            },
            "post_processing_config": {
                "representation": representation,
                "representation_description": {
                    "Surface": "实体表面渲染，显示物体的外表面",
                    "Surface With Edges": "实体表面渲染，同时显示网格边界线",
                    "Wireframe": "线框模式，仅显示网格结构",
                    "Points": "点云模式，仅显示网格节点"
                }.get(representation, ""),
                "background": background,
                "show_scalar_bar": show_scalar_bar,
                "camera_rotation_degrees": camera_rotation,
                "camera_rotation_axis": camera_axis,
                "image_resolution": f"{width}x{height}",
            },
            "images": metadata_list,
        }
        
        json_path = target_path.parent / f"{target_path.stem}_info.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"[VTUScreenshot] Generated {len(generated_paths)} screenshots in: {target_path.parent}")
        print(f"[VTUScreenshot] Files: {[Path(p).name for p in generated_paths]}")
        print(f"[VTUScreenshot] Metadata: {json_path.name}")
        
        # Return the directory path where all images are saved
        output_dir = str(target_path.parent)
        return output_dir


class TO_VTK(CNodeType):
    r"""Convert simulation results to VTK format for visualization.

    Inputs:
        mesh (mesh): Computational mesh.
        uh (tensor): Numerical solution vector.
        path (string): Output directory to store the VTK file.

    Outputs:
        path (string): Directory where the VTK file is written.
    """

    TITLE: str = "导出VTK文件"
    PATH: str = "后处理.导出VTK"
    DESC: str = "将模拟结果导出为VTK格式文件，便于使用可视化工具进行后续分析与展示。"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("uh", DataType.TENSOR, title="数值解"),
        PortConf("path", DataType.STRING, title="导出路径"),
    ]
    OUTPUT_SLOTS = [
        PortConf("path", DataType.STRING, title="导出路径"),
    ]

    @staticmethod
    def run(mesh, uh, path):
        try:
            uh_len = len(uh)  # type: ignore[arg-type]
        except Exception:
            uh_len = None
        print("[TO_VTK] inputs", {"path": path, "uh_len": uh_len})
        from pathlib import Path

        export_dir = Path(path).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        mesh.nodedata["uh"] = uh
        fname = export_dir / "test.vtu"
        mesh.to_vtk(fname=str(fname))
        print(f"[TO_VTK] wrote {fname}")

        return str(fname.resolve())


def _parse_background(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse and validate background color specification."""
    presets = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
    }

    if isinstance(value, str):
        text = value.strip().lower()
        if text in presets:
            return presets[text]
        if text.startswith("#") and len(text) == 7:
            r = int(text[1:3], 16) / 255.0
            g = int(text[3:5], 16) / 255.0
            b = int(text[5:7], 16) / 255.0
            return _clamp_rgb((r, g, b))
        parts = [p for p in text.replace(";", " ").replace(",", " ").split(" ") if p]
        values = [float(p) for p in parts]
    else:
        values = [float(x) for x in value]

    if not values:
        raise ValueError("background colour definition is empty.")
    if len(values) == 1:
        values *= 3
    if len(values) < 3:
        raise ValueError("background colour requires at least three components.")

    return _clamp_rgb(tuple(values[:3]))


def _array_range(data_attr, array_name: str) -> tuple[float, float]:
    """Get data range for an array."""
    array = data_attr.GetArray(array_name)
    if array is None:
        return 0.0, 1.0
    rng = array.GetRange(-1)
    if not isinstance(rng, Iterable) or len(rng) < 2:
        return 0.0, 1.0
    lower, upper = rng[:2]
    if lower == upper:
        eps = max(abs(lower), 1.0) * 1e-6
        return lower - eps, upper + eps
    return float(lower), float(upper)


def _clamp_rgb(rgb: Sequence[float]) -> tuple[float, float, float]:
    """Clamp RGB values to [0, 1] range."""
    return tuple(max(0.0, min(1.0, float(component))) for component in rgb[:3])


def _ensure_point_array(dataset: object, array_name: str, location: str) -> object:
    if dataset is None or location != "POINTS":
        return dataset

    if not hasattr(dataset, "GetPointData") or not hasattr(dataset, "GetCellData"):
        return dataset

    point_data = dataset.GetPointData()
    if point_data is not None and point_data.HasArray(array_name) == 1:
        return dataset

    cell_data = dataset.GetCellData()
    if cell_data is None or cell_data.HasArray(array_name) != 1:
        return dataset

    try:
        from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
    except ModuleNotFoundError:
        return dataset

    converter = vtkCellDataToPointData()
    converter.SetInputData(dataset)
    converter.PassCellDataOn()
    converter.Update()

    converted = converter.GetOutput()
    if converted is None or not hasattr(converted, "NewInstance"):
        return dataset

    clone = converted.NewInstance()
    clone.DeepCopy(converted)

    for attr_name in ("__fealpy_vtu_path", "__fealpy_active_array"):
        attr_val = getattr(dataset, attr_name, None)
        if attr_val is not None:
            try:
                setattr(clone, attr_name, attr_val)
            except AttributeError:
                pass

    return clone
