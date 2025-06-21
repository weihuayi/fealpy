
class Quadrature():
    def number_of_quadrature_points(self):
        return self.quadpts.shape[0]

    def get_quadrature_points_and_weights(self):
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i):
        return self.quadpts[:, i], self.weights[i]
