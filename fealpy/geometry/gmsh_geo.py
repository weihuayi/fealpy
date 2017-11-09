'''
This is a helper class to write gmsh geo
'''
import copy

class Point(object):
    _ID = 0
    def __init__(self, x, cl):
        self.x = x
        self.clength = cl 

        self.id = 'p{}'.format(Point._POINT_ID)
        Point._ID += 1

        # Points are always 3D in gmsh
        if len(x) == 2:
            self.code = '\n'.join([
                '{} = newp;'.format(self.id),
                'Point({}) = {{{!r}, {!r}, {!r}, {!r}}};'.format(
                    self.id, x[0], x[1], 0.0, cl 
                )])
        elif len(x) == 3:
            self.code = '\n'.join([
                '{} = newp;'.format(self.id),
                'Point({}) = {{{!r}, {!r}, {!r}, {!r}}};'.format(
                    self.id, x[0], x[1], x[2], cl 
                )])
        return

class LineBase(object):
    _ID = 0
    def __init__(self, id0=None):
        if id0:
            self.id = id0
        else:
            self.id = 'l{}'.format(LineBase._ID)
            LineBase._ID += 1
        return

    def __neg__(self):
        neg_self = copy.deepcopy(self)
        neg_self.id = '-' + neg_self.id
        return neg_self

class Line(LineBase):
    def __init__(self, p0, p1):
        super(Line, self).__init__()
        assert isinstance(p0, Point)
        assert isinstance(p1, Point)
        self.points = [p0, p1]

        self.code = '\n'.join([
            '{} = newl;'.format(self.id),
            'Line({}) = {{{}, {}}};'.format(self.id, p0.id, p1.id)
            ])
        return

class LineLoop(object):
    _ID = 0

    def __init__(self, lines):
        self.lines = lines

        self.id = 'll{}'.format(LineLoop._ID)
        LineLoop._ID += 1

        self.code = '\n'.join([
            '{} = newll;'.format(self.id),
            'Line Loop({}) = {{{}}};'.format(
                self.id, ', '.join([l.id for l in lines])
            )])
        return

    def __len__(self):
        return len(self.lines)

class Surface(object):
    _ID = 0
    num_edges = 0

    def __init__(self, line_loop, api_level=2):
        assert isinstance(line_loop, LineLoop)

        self.line_loop = line_loop

        self.id = 'rs{}'.format(Surface._ID)
        Surface._ID += 1

        self.code = '\n'.join([
            '{} = news;'.format(self.id),
            '{}({}) = {{{}}};'.format('Surface', self.id, self.line_loop.id)
            ])
        self.num_edges = len(line_loop)
        return

class SurfaceBase(object):
    _ID = 0
    num_edges = 0

    def __init__(self, id0=None, num_edges=0):
        isinstance(id0, str)
        if id0:
            self.id = id0
        else:
            self.id = 's{}'.format(SurfaceBase._ID)
            SurfaceBase._ID += 1
        self.num_edges = num_edges
        return

class PlaneSurface(SurfaceBase):
    def __init__(self, line_loop, holes=None):
        super(PlaneSurface, self).__init__()

        assert isinstance(line_loop, LineLoop)
        self.line_loop = line_loop

        if holes is None:
            holes = []

        # The input holes are either line loops or entities that contain line
        # loops (like polygons).
        self.holes = [
            h if isinstance(h, LineLoop) else h.line_loop
            for h in holes
            ]

        line_loops = [self.line_loop] + self.holes
        self.code = '\n'.join([
            '{} = news;'.format(self.id),
            'Plane Surface({}) = {{{}}};'.format(
                self.id, ','.join([ll.id for ll in line_loops])
            )])
        self.num_edges = len(self.line_loop) + sum(len(h) for h in self.holes)
        return


class GmeshGeo(Object):
    def __init__(self):
        self._GMSH_CODE = ['// This code was created by fealpy']

    def get_code(self):
        '''Returns properly formatted Gmsh code.
        '''
        return '\n'.join(self._GMSH_CODE)

    ## add geometry objects functions
    def add_comment(self, string):
        self._GMSH_CODE.append('// ' + string)
        return

    def add_point(self, x, cl):
        p = Point(x, cl)
        self._GMSH_CODE.append(p.code)
        return p

    def add_points(self, x, cl):
        return [self.add_point(p, l) for p, l in zip(x, cl)]

    def add_line(self, p0, p1):
        p = Line(p0, p1)
        self._GMSH_CODE.append(p.code)
        return p

    def add_surface(self, line_loop):
        s = Surface(line_loop)
        self._GMSH_CODE.append(s.code)
        return s

    def add_point_in_surface(self, point, surface):
        code = 'Point{{{}}} In Surface{{{}}};'.format(point.id, surface.id)
        print(code)
        self._GMSH_CODE.append(code)

    def add_line_in_surface(self, line, surface):
        code = 'Line{{{}}} In Surface{{{}}};'.format(line.id, surface.id)
        self._GMSH_CODE.append(code)

    def add_plane_surface(self, *args, **kwargs):
        p = PlaneSurface(*args, **kwargs)
        self._GMSH_CODE.append(p.code)
        return p

    def add_polygon(self, points, cl, holes=None, make_surface=True):
        if holes is None:
            holes = []
        else:
            assert make_surface

        # Create points.
        p = self.add_points(points, cl)
        # Create lines
        lines = [self.add_line(p[k], p[k+1]) for k in range(len(p)-1)]
        lines.append(self.add_line(p[-1], p[0]))
        ll = self.add_line_loop((lines))
        surface = self.add_plane_surface(ll, holes) if make_surface else None

        class Polygon(object):
            def __init__(self, line_loop, surface, lcar):
                self.line_loop = line_loop
                self.surface = surface
                self.lcar = lcar
                return

        return Polygon(ll, surface, lcar)

    ## algorithm
    def set_mesh_algorithm(self, alg):
        code = 'Mesh.Algorithm={};'.format(alg)
        self._GMSH_CODE.append(code)
