import numpy as np
from ctypes import c_void_p
from PIL import Image
from OpenGL.GL import *
from OpenGL.arrays import vbo

from fealpy import logger
import ipdb
import time
import os

class GLMesh:
    def __init__(self, node, cell=None, texture_paths=[], 
                 texture_folders=[],
                 texture_unit=0,
                 flip='LR0', tag = 'single', shader_program = None):
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
        self.texture_paths = texture_paths
        self.texture_folders = texture_folders
        self.texture_ids = None
        self.texture_unit = texture_unit # TODO 作用?
        self.shader_program = shader_program

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
            # 解释每个参数的意义
            # 0: 顶点属性的索引
            # 3: 每个顶点属性的大小
            # GL_FLOAT: 数据类型
            # GL_FALSE: 是否需要归一化
            # 5 * self.node.itemsize: 步长
            # c_void_p(0): 偏移量
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

        elif self.node.shape[1] == 8:  # Positions, uv0, uv1, and weight
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * self.node.itemsize, c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * self.node.itemsize, c_void_p(3 * self.node.itemsize))

            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * self.node.itemsize, c_void_p(5 * self.node.itemsize))

            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * self.node.itemsize, c_void_p(7 * self.node.itemsize))

        # Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # Load texture if provided
        if len(self.texture_paths) > 0:
            images = [self.get_image(path) for path in self.texture_paths]
            self.texture_ids = [self.load_texture(image) for image in images]
        if len(self.texture_paths) > 1:
            print(self.texture_ids)
            print(self.texture_paths)
            #self.load_texture()

    def get_all_textures(self):
        images_path = []
        # self.texture_paths
        # 是一个文件夹的列表，现在取出其中的每个文件夹中的图片的路径
        for path in self.texture_folders:
            images_path.append(sorted(os.listdir(path)))

        NF = len(images_path)  # Number of folders
        NP = len(images_path[0]) # Number of pictures
        NP = 10
        images = []
        for i in range(NP):
            image0 = []
            for j in range(NF):
                img = self.get_image(self.texture_folders[j] + '/' + images_path[j][i])
                image0.append(img)
            images.append(image0)
        print("Read all images successfully!")
        return images

    def get_folder_textures(self, folder):
        images_path = sorted(os.listdir(folder))

        NP = len(images_path) # Number of pictures
        NP = 500
        images = []
        for i in range(NP):
            img = self.get_image(folder + '/' + images_path[i])
            images.append(img)
        print("Read all images successfully!")
        return images

    def get_image(self, image_path):
        image = Image.open(image_path)
        if self.flip == 'TB':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image for OpenGL
        elif self.flip == 'LR':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image for OpenGL
        img_data = image.convert("RGBA").tobytes()  # Convert the image to RGBA format
        return (img_data, image.width, image.height)

    def load_texture(self, image_data, texture_id = None):
        """
        @brief 加载并配置网格的纹理。

        纹理图片从指定的路径加载。加载后的纹理将绑定到纹理单元并配置相应的纹理参数。
        """
        # Generate a texture ID and bind it
        if texture_id is None:
            texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        img_data, width, height = image_data

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load the image data into the texture object
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        # Generate mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    def draw_face(self):
        """
        @brief 画面
        """
        shader_program = self.shader_program
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 0)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))

    def draw_edge(self):
        """
        @brief 画边
        """
        shader_program = self.shader_program
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # 绘制线框
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 恢复默认模式

    def bind_texture(self, texture_id, i):
        """
        @brief 绑定纹理
        """
        shader_program = self.shader_program
        glActiveTexture(GL_TEXTURE0 + self.texture_unit + i)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        loc = glGetUniformLocation(shader_program, "textureSampler"+str(i))
        glUniform1i(loc, self.texture_unit+i)

        #value = np.zeros(1, dtype=np.int32)
        #glGetUniformiv(shader_program, loc, value)

    def draw_texture(self):
        """
        @brief 显示纹理
        """
        shader_program = self.shader_program
        glUniform1i(glGetUniformLocation(shader_program, "mode"), 3)

        for i, texture_id in enumerate(self.texture_ids):
            self.bind_texture(texture_id, i)

        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glDrawElements(GL_TRIANGLES, len(self.cell), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(self.node))

    def draw(self, mode):
        """
        @brief 使用提供的着色器程序绘制网格。

        @param shader_program: 用于绘制网格的着色器程序ID。
        @param mode: 显示模式控制

        该方法绑定网格的VAO和纹理（如果有），并根据是否提供了单元格索引来执行绘制命令。
        """
        glBindVertexArray(self.vao)  # Bind the VAO for this mesh
        glUseProgram(self.shader_program)


        if mode == 3:
            if self.texture_ids is not None and self.node.shape[1] in [5, 8]: 
                self.draw_texture()
            else:
                self.draw_face()
        elif mode == 2:
            self.draw_edge() # 先画边，后画面
            self.draw_face()
        elif mode == 1:
            self.draw_edge()
        elif mode == 0:
            self.draw_face()

        glBindVertexArray(0)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        if len(self.texture_paths) > 0:
            glBindTexture(GL_TEXTURE_2D, 0)


    def redraw(self, mode, images):
        """
        @brief 使用提供的着色器程序绘制网格。

        @param shader_program: 用于绘制网格的着色器程序ID。
        @param mode: 显示模式控制

        该方法绑定网格的VAO和纹理（如果有），并根据是否提供了单元格索引来执行绘制命令。
        """
        self.texture_ids = [self.load_texture(img, tid) for img, tid in zip(images, self.texture_ids)]

        glBindVertexArray(self.vao)  # Bind the VAO for this mesh
        glUseProgram(self.shader_program)

        if mode == 3:
            if self.texture_ids is not None and self.node.shape[1] in [5, 8]: 
                self.draw_texture()
            else:
                self.draw_face()
        elif mode == 2:
            self.draw_edge() # 先画边，后画面
            self.draw_face()
        elif mode == 1:
            self.draw_edge()
        elif mode == 0:
            self.draw_face()

        glBindVertexArray(0)
        if self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        if len(self.texture_paths) > 0:
            glBindTexture(GL_TEXTURE_2D, 0)















