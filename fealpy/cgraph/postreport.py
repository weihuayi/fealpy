from typing import Union
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["SolidReport"]


class SolidReport(CNodeType):
    r"""Solid Mechanics Computation Report Node.
    
    Inputs:
        mesh (MESH): The computational mesh object used in the simulation.
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
    DESC: str = "生成包含网格、材料与计算结果的固体力学分析报告"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="计算网格对象", title="网格"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的弹性模量"),
        PortConf("beam_mu", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的剪切模量"),
        PortConf("Ax", DataType.TENSOR, 1, desc="横截面积",  title="X 方向的横截面积"),
        PortConf("Ay", DataType.TENSOR, 1, desc="横截面积",  title="Y 方向的横截面积"),
        PortConf("Az", DataType.TENSOR, 1, desc="横截面积",  title="Z 方向的横截面积"),
        PortConf("J", DataType.TENSOR, 1, desc="极性矩",  title="极性矩"),
        PortConf("Iy", DataType.TENSOR, 1, desc="惯性矩",  title="Y 轴的惯性矩"),
        PortConf("Iz", DataType.TENSOR, 1, desc="惯性矩",  title="Z 轴的惯性矩"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的弹性模量"),
        PortConf("axle_mu", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的剪切模量"),
        PortConf("uh", DataType.TENSOR, 1, desc="位移结果", title="位移数值解"),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("report", DataType.NONE, title="报告文件(PDF)")  
    ]

    @staticmethod
    def run(**options):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
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
        
        filename = "solid_report.pdf"
        filepath = os.path.abspath(filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        mesh = options.get("mesh")
        uh = options.get("uh")
        disp = uh.reshape(-1, 6)
        
        material_info = {
            "梁的弹性模量": options.get("beam_E"),
            "梁的泊松比": options.get("beam_mu"),
            "X 方向的横截面积": options.get("Ax"),
            "Y 方向的横截面积": options.get("Ay"),
            "Z 方向的横截面积": options.get("Az"),
            "极性矩": options.get("J"),
            "Y 轴的惯性矩": options.get("Iy"),
            "Z 轴的惯性矩": options.get("Iz"),
            "弹簧的弹性模量": options.get("axle_E"),
            "弹簧的剪切模量": options.get("axle_mu"),
        } 
        
        mesh_info = {
            "节点的个数": getattr(mesh, "number_of_nodes", lambda: "NN")(),
            "单元的个数": getattr(mesh, "number_of_cells", lambda: "NC")(),
        }
        
        # Title
        story.append(Paragraph("<b>列车轮轴场景计算结果报告</b>", styles["Title"]))
        story.append(Spacer(1, 0.5*cm))
        
        story.append(Paragraph("<b>一、 网格信息</b>", styles["Heading2"]))
        for k, v in mesh_info.items():
            story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))
        
        
        story.append(Paragraph("<b>二、 材料属性</b>", styles["Heading2"]))
        data = [["属性名称", "数值"]]
        for k, v in material_info.items():
            value_text = Paragraph(str(v).replace("\n", "<br/>"), styles["Normal"])
            data.append([Paragraph(k, styles["Normal"]), value_text])
        table = Table(data, hAlign="LEFT", colWidths=[4*cm, 12*cm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.4*cm))
        
        
        story.append(Paragraph("<b>三、 计算结果</b>", styles["Heading2"]))
        story.append(Paragraph("<b>3.1 位移</b>", styles["Heading3"]))
        story.append(Paragraph(f"Displacement (u): {disp}", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))
        
        doc.build(story)
        print(f"✅ PDF report generated: {filepath}")
        
        return {"report": filepath}