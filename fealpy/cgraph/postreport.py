from typing import Union
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["SolidReport"]


class SolidReport(CNodeType):
    r"""Solid Mechanics Computation Report Node.
    
    Inputs:
        path(string): 
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        section_shapes (MENU): Beam cross-section shape configuration parameter.
        shear_factors (FLOAT): Shear correction factor. Default 10/9.
        mesh (MESH): The computational mesh object used in the simulation.
        property (string): Material type.
        beam_E (FLOAT): Young’s modulus of the beam.
        beam_mu (FLOAT): Shear modulus of the beam.
        Ax, Ay, Az (TENSOR): Cross-sectional areas in X, Y, and Z directions.
        J (TENSOR): Polar moment of inertia.
        Iy, Iz (TENSOR): Moments of inertia about Y and Z axes.
        axle_E (FLOAT): Young’s modulus of the spring (axle).
        axle_mu (FLOAT): Shear modulus of the spring (axle).
        uh (TENSOR): Displacement field results.
        
    Outputs:
        report (FILE): Path to the generated PDF report.
    
    """
    TITLE: str = "固体力学计算报告"
    PATH: str = "后处理.报告生成"
    DESC: str = "生成包含几何、网格、材料与计算结果信息的固体力学分析报告"
    INPUT_SLOTS = [
        PortConf("path", DataType.STRING, 1, desc="仿真报告的存储路径", title="存储路径"),
        PortConf("beam_para", DataType.TENSOR, 0, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 0, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("section_shapes", DataType.MENU, 0, desc="梁的截面形状", title="梁截面形状", default="circular", 
                 items=["circular", "rectangular", "I-shaped", "H-shaped"]),
        PortConf("shear_factors",DataType.FLOAT, 0, desc="梁剪切变形计算中的修因子，圆截面推荐值为 10/9",
                 title="剪切修正因子", param="kappa", default=10/9), 
        PortConf("mesh", DataType.MESH, 1, desc="计算网格对象", title="网格"),
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="轮轴材料类型选择", title="梁材料", default="Timo_beam", 
                 items=["Euler_beam", "Timo_beam"]),
        PortConf("axle_type", DataType.STRING, 0, desc="轮轴材料类型", title="弹簧材料", default="Spring"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的弹性模量"),
        PortConf("beam_mu", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的剪切模量"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的弹性模量"),
        PortConf("axle_mu", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的剪切模量"),
        PortConf("uh", DataType.TENSOR, 1, desc="位移结果", title="位移数值解")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("report", DataType.NONE, title="报告文件路径")
    ]

    @staticmethod
    def run(**options):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (Paragraph, 
                                    SimpleDocTemplate, 
                                    Spacer, Table, TableStyle, Image)
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        import os
        
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        styles = getSampleStyleSheet()
        for name in styles.byName:
            styles[name].fontName = "STSong-Light"
        
        path = options.get("path")
        file = "report"
        filename ="列车轮轴场景仿真结果报告.pdf"
        folder = os.path.join(path, file)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        mesh = options.get("mesh")
        uh = options.get("uh")
        disp = uh.reshape(-1, 6)
        
        geometry_info = {
            "梁段参数 [直径, 长度, 数量]": options.get("beam_para"),
            "轴段参数 [直径, 长度, 数量]": options.get("axle_para"),
            "梁的截面形状": options.get("section_shapes"),
            "剪切修正因子": options.get("kappa"),
        }
        
        mesh_info = {
            "节点的个数": getattr(mesh, "number_of_nodes", lambda: "NN")(),
            "单元的个数": getattr(mesh, "number_of_cells", lambda: "NC")(),
        }
        
        
        material_info = {
            "材料材质": options.get("property"),
            "梁材料类型": options.get("beam_type"),
            "弹簧材料类型": options.get("axle_type"),
            "梁的弹性模量": options.get("beam_E"),
            "梁的泊松比": options.get("beam_mu"),
            "弹簧的弹性模量": options.get("axle_E"),
            "弹簧的剪切模量": options.get("axle_mu"),
        } 
        
        # Title
        story.append(Paragraph("<b>列车轮轴场景仿真结果报告</b>", styles["Title"]))
        story.append(Spacer(1, 0.5*cm))
        
        story.append(Paragraph("<b>一、 几何结构信息</b>", styles["Heading2"]))

        # geometry data
        geometry_data = [["参数名称", "数值"]]
        for k, v in geometry_info.items():
            v_text = Paragraph(str(v).replace("\n", "<br/>"), styles["Normal"])
            geometry_data.append([Paragraph(k, styles["Normal"]), v_text])

        geometry_table = Table(geometry_data, hAlign="LEFT", colWidths=[5*cm, 11*cm])
        geometry_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),  # 表头灰色
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(geometry_table)
        story.append(Spacer(1, 0.4*cm))
        
        story.append(Paragraph("<b>二、 网格信息</b>", styles["Heading2"]))
        story.append(Paragraph("<b>2.1 网格节点和单元数</b>", styles["Heading3"]))
        mesh_data = [["网格信息", "数值"]]
        for k, v in mesh_info.items():
            mesh_data.append([Paragraph(k, styles["Normal"]), Paragraph(str(v), styles["Normal"])])
            
        mesh_table = Table(mesh_data, hAlign="LEFT", colWidths=[4*cm, 12*cm])
        mesh_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),  # 中文字体
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),  # 表头灰底
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(mesh_table)
        story.append(Spacer(1, 0.4*cm))
     
        story.append(Paragraph("<b>2.2 网格图</b>", styles["Heading3"]))
        
        
        story.append(Paragraph("<b>三、 材料属性</b>", styles["Heading2"]))
        data = [["属性名称", "数值"]]
        for k, v in material_info.items():
            value_text = Paragraph(str(v).replace("\n", "<br/>"), styles["Normal"])
            data.append([Paragraph(k, styles["Normal"]), value_text])
        table = Table(data, hAlign="LEFT", colWidths=[4*cm, 12*cm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),                # 中文字体
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),   # 仅表头灰色
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),                 # 表格边框
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.4*cm))

        
        story.append(Paragraph("<b>四、 计算结果</b>", styles["Heading2"]))
        story.append(Paragraph("<b>4.1 位移数值</b>", styles["Heading3"]))
        columns = ["Node", "u", "v", "w", "theta_x", "theta_y", "theta_z"]
        table_data = [columns]
        for i in range(disp.shape[0]):
            row = [str(i)] + [f"{val:.6e}" for val in disp[i]]
            table_data.append(row)
        
        table = Table(table_data, hAlign="LEFT", repeatRows=1,
              colWidths=[1.5*cm] + [2.5*cm]*6)
        
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),  
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),  # 表头灰底
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))
        
        story.append(Paragraph("<b>4.2 位移云图</b>", styles["Heading3"]))
        
        doc.build(story)
        print(f"PDF report generated: {filepath}")
        
        return {"report": filepath}