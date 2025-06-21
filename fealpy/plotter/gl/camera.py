import glm
import numpy as np

class Camera:
    def __init__(self, 
            pos=(0.0, 0.0, 3.0), 
            front=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0)):
        self.camera_pos = glm.vec3(pos[0], pos[1], pos[2])
        self.camera_front = glm.vec3(front[0], front[1], front[2])
        self.camera_up = glm.vec3(up[0], up[1], up[2])
        self.yaw = -90.0
        self.pitch = 0.0
        self.lastX = 400
        self.lastY = 300
        self.first_mouse = True
        self.fov = 45.0

    def get_view_matrix(self):
        return glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_mouse_movement(self, xpos, ypos, constrain_pitch=True):
        if self.first_mouse:
            self.lastX = xpos
            self.lastY = ypos
            self.first_mouse = False

        xoffset = xpos - self.lastX
        yoffset = self.lastY - ypos  # reversed since y-coordinates range from bottom to top
        self.lastX = xpos
        self.lastY = ypos

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # Make sure that when pitch is out of bounds, screen doesn't get flipped
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        direction = glm.vec3()
        direction.x = np.cos(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        direction.y = np.sin(glm.radians(self.pitch))
        direction.z = np.sin(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        self.camera_front = glm.normalize(direction)

    # This method can be extended to handle keyboard input
    def process_keyboard(self, direction, delta_time):
        pass

