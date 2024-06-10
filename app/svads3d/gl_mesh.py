import numpy as np
from ctypes import c_void_p
from PIL import Image
from OpenGL.GL import *
from OpenGL.arrays import vbo

from fealpy import logger
import ipdb

class GLMesh:
    def __init__(self, node, cell=None, texture_path=None, texture_unit=0, flip='LR'):
        """
        @brief 初始化网格类，根据节点的数据格式配置顶点属性，并加载纹理（如果提供）。

        @param node: 节点数组，其形状可以是 (NN, 3), (NN, 5), 或 (NN, 6)。
                     (NN, 3) 仅包含顶点位置，
                     (NN, 5) 包含顶点位置和纹理坐标，
                     (NN, 6) 包含顶点位置、纹理坐标和一个法线信息。
        @param cell: 单元格的索引数组，用于绘制网格。如果为None，则使用顶点数组直接绘制。
        @param texture_path: 纹理图片的路径。如果为None，则不加载纹理。
        """
        self.flip = flip
        self.node = node 
        self.cell = cell if cell is not None else None
        self.texture_path = texture_path
        self.texture_id = None
        self.texture_unit = texture_unit

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = None if self.cell is None else glGenBuffers(1)

        # Bind the VAO
        glBindVertexArray(self.vao)

        # Bind and set VBO with node data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.node.nbytes, self.node, GL_STATIC_DRAW)

        # If cell data is provided, bind and set EBO
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.cell.nbytes, self.cell, GL_STATIC_DRAW)

        # 根据 node 数组的列数设置顶点的属性
        if self.node.shape[1] == 3:  # Only positions
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * self.node.itemsize, c_void_p(0))
        elif self.node.shape[1] == 5:  # Positions and texture coordinates
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * self.node.itemsize, c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * self.node.itemsize, c_void_p(3 * self.node.itemsize))

        elif self.node.shape[1] == 6:  # Positions, texture coordinates, and normals
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * self.node.itemsize, c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * self.node.itemsize, c_void_p(3 * self.node.itemsize))

            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 6 * self.node.itemsize, c_void_p(5 * self.node.itemsize))

        # Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # Load texture if provided
        if self.texture_path:
            self.load_texture()

    def load_texture(self):
        """
        @brief 加载并配置网格的纹理。

        纹理图片从指定的路径加载。加载后的纹理将绑定到纹理单元并配置相应的纹理参数。
        """
        # Load the image with Pillow
        image = Image.open(self.texture_path)
        if self.flip == 'TB':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image for OpenGL
        elif self.flip == 'LR':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image for OpenGL
        img_data = image.convert("RGBA").tobytes()  # Convert the image to RGBA format

        # Generate a texture ID and bind it
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load the image data into the texture object
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        # Generate mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_face(self, shader_program):
        """
        @brief 画面
        """
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 0)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))

    def draw_edge(self, shader_program):
        """
        @brief 画边
        """
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # 绘制线框
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 恢复默认模式

    def draw_texture(self, shader_program):
        """
        @brief 显示纹理
        """
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 3)
        glActiveTexture(GL_TEXTURE0 + self.texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glUniform1i(glGetUniformLocation(shader_program, "textureSampler"), self.texture_unit)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))


    def draw(self, shader_program, mode):
        """
        @brief 使用提供的着色器程序绘制网格。

        @param shader_program: 用于绘制网格的着色器程序ID。
        @param mode: 显示模式控制

        该方法绑定网格的VAO和纹理（如果有），并根据是否提供了单元格索引来执行绘制命令。
        """
        glBindVertexArray(self.vao)  # Bind the VAO for this mesh

        if mode == 3:
            if self.texture_id is not None and self.node.shape[1] == 5: 
                self.draw_texture(shader_program)
            else:
                self.draw_face(shader_program)
        elif mode == 2:
            self.draw_edge(shader_program) # 先画边，后画面
            self.draw_face(shader_program)
        elif mode == 1:
            self.draw_edge(shader_program)
        elif mode == 0:
            self.draw_face(shader_program)

        glBindVertexArray(0)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        if self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, 0)


