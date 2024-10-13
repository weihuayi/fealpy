'''
This is a helper class to write gmsh geo
'''
import copy

class Point(object):
    _ID = 0
    def __init__(self, x, cl):
        self.x = x
        self.clength = cl 

        self.id = 'p{}'.format(Point._ID)
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
    def __init__(self, line_loop):
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
            h if isinstance(h, LineLoop) else h.line_loop for h in holes
            ]

        line_loops = [self.line_loop] + self.holes
        self.code = '\n'.join([
            '{} = news;'.format(self.id),
            'Plane Surface({}) = {{{}}};'.format(
                self.id, ','.join([ll.id for ll in line_loops])
            )])
        self.num_edges = len(self.line_loop) + sum(len(h) for h in self.holes)
        return

class FieldBase():
    _ID = 0
    def __init__(self):
        self.id = 'field{}'.format(FieldBase._ID)
        FieldBase._ID += 1

#            'Field[{}].NodesList = {{{}}};'.format(self.id,
#                ','.join([p.id for p in nodes])
#                ),

class AttractorField(FieldBase):
    def __init__(self, points, edges):
        super(EdgesAttractorField, self).__init__()
        self.code ='\n'.join([
            '{} = newf;'.format(self.id),
            'Field[{}] = Attractor;'.format(self.id),
            'Field[{}].EdgesList = {{{}}};'.format(self.id, 
                ','.join([e.id for e in edges])),
            'Field[{}].NodesList = {{{}}};'.format(self.id, 
                ','.join([p.id for p in points])
                )
            ])

class ThresholdField(FieldBase):
    def __init__(self, ifield, clmin, clmax, distmin, distmax):
        super(ThresholdField, self).__init__()
        self.code = '\n'.join([
            '{} = newf;'.format(self.id),
            'Field[{}] = Threshold;'.format(self.id),
            'Field[{}].IField = {};'.format(self.id, ifield.id),
            'Field[{}].LcMin = {};'.format(self.id, clmin),
            'Field[{}].LcMax = {};'.format(self.id, clmax),
            'Field[{}].DistMin = {};'.format(self.id, distmin),
            'Field[{}].DistMax = {};'.format(self.id, distmax)
            ])
        return 

class BoundaryLayerField(FieldBase):
    def __init__(self, points, edges, hfar, hwall_n, ratio, thickness):
        super(BoundaryLayerField, self).__init__()
        self.code = '\n'.join([
            '{} = newf;'.format(self.id),
            'Field[{}] = BoundaryLayer;'.format(self.id),
            'Field[{}].NodesList = {{{}}};'.format(self.id, 
                ','.join([p.id for p in points])),
            'Field[{}].EdgesList = {{{}}};'.format(self.id, 
                ','.join([e.id for e in edges])),
            'Field[{}].hfar = {{{}}};'.format(self.id, hfar),
            'Field[{}].hwall_n = {{{}}};'.format(self.id, hwall_n),
            'Field[{}].ratio = {{{}}};'.format(self.id, ratio),
            'Field[{}].thickness = {{{}}};'.format(self.id, thickness)
            ])
        return 


class MinField(FieldBase):
    def __init__(self, fields):
        super(MinField, self).__init__()
        self.code = '\n'.join([
            '{} = newf;'.format(self.id),
            'Field[{}] = Min;'.format(self.id),
            'Field[{}].FieldsList = {{{}}};'.format(self.id,
                ','.join([f.id for f in fields])
                )
            ])

class GmshGeo(object):
    def __init__(self):
        self._TAKEN_PHYSICALGROUP_IDS = []
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

    def add_points(self, ps, cl):
        return [self.add_point(p, cl) for p in ps]

    def add_line(self, p0, p1):
        p = Line(p0, p1)
        self._GMSH_CODE.append(p.code)
        return p

    def add_line_loop(self, lines):
        p = LineLoop(lines)
        self._GMSH_CODE.append(p.code)
        return p

    def add_surface(self, line_loop):
        s = Surface(line_loop)
        self._GMSH_CODE.append(s.code)
        return s

    def add_segments_in_surface(self, points, segments, surface):
        lines = [self.add_line(points[seg[0]], points[seg[1]]) for seg in segments]
        for line in lines:
            self.add_line_in_surface(line, surface)
        return lines

    def add_point_in_surface(self, point, surface):
        code = 'Point{{{}}} In Surface{{{}}};'.format(point.id, surface.id)
        self._GMSH_CODE.append(code)

    def add_line_in_surface(self, line, surface):
        code = 'Line{{{}}} In Surface{{{}}};'.format(line.id, surface.id)
        self._GMSH_CODE.append(code)

    def add_plane_surface(self, line_loop, holes=None):
        ps = PlaneSurface(line_loop, holes=holes)
        self._GMSH_CODE.append(ps.code)
        return ps

    def add_rectangle(self, box, cl, z=0.0, holes=None, make_surface=True):
        return self.add_polygon([
                [box[0], box[2], z],
                [box[1], box[2], z],
                [box[1], box[3], z],
                [box[0], box[3], z]
                ],
                cl,
                holes=holes,
                make_surface=make_surface
                )

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
            def __init__(self, line_loop, surface, cl):
                self.line_loop = line_loop
                self.surface = surface
                self.clength = cl 
                return

        return Polygon(ll, surface, cl)

    ## field
    def add_attractor_field(self, edges):
        field = AttractorField(edges)
        self._GMSH_CODE.append(field.code)
        return field

    def add_threshold_field(self, ifield, clmin, clmax, distmin, distmax):
        field = ThresholdField(ifield, clmin, clmax, distmin, distmax)
        self._GMSH_CODE.append(field.code)
        return field

    def add_boundarylayer_field(self, points, edges, 
            hfar=1.0, 
            hwall_n=0.01,
            ratio=1.1,
            thickness = 0.03):
        field = BoundaryLayerField(points, edges, hfar, hwall_n, ratio, thickness)
        self._GMSH_CODE.append(field.code)
        return field

    def add_min_field(self, fields):
        field = MinField(fields)
        self._GMSH_CODE.append(field.code)
        return field

    def set_background_field(self, field):
        code = 'Background Field = {};'.format(field.id)
        self._GMSH_CODE.append(code)
        return 

    ## physical group

    def _new_physical_group(self, label=None):
        # See
        # https://github.com/nschloe/pygmsh/issues/46#issuecomment-286684321
        # for context.
        max_id = \
            0 if not self._TAKEN_PHYSICALGROUP_IDS  \
            else max(self._TAKEN_PHYSICALGROUP_IDS)

        if label is None:
            label = max_id + 1

        if isinstance(label, int):
            assert label not in self._TAKEN_PHYSICALGROUP_IDS
            self._TAKEN_PHYSICALGROUP_IDS += [label]
            return str(label)

        assert _is_string(label)
        self._TAKEN_PHYSICALGROUP_IDS += [max_id + 1]
        return '"{}"'.format(label)

    def _add_physical(self, tpe, entities, label=None):
        label = self._new_physical_group(label)
        if not isinstance(entities, list):
            entities = [entities]
        self._GMSH_CODE.append(
            'Physical {}({}) = {{{}}};'.format(
                tpe, label, ', '.join([e.id for e in entities])
            ))
        return

    def add_physical_point(self, points, label=None):
        self._add_physical('Point', points, label=label)
        return

    def add_physical_line(self, lines, label=None):
        self._add_physical('Line', lines, label=label)
        return

    def add_physical_surface(self, surfaces, label=None):
        self._add_physical('Surface', surfaces, label=label)
        return

    ## algorithm
    def set_mesh_algorithm(self, alg):
        code = 'Mesh.Algorithm={};'.format(alg)
        self._GMSH_CODE.append(code)
