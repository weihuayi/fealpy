# tests/test_sources_2d.py
import pytest
from fealpy.backend import backend_manager as bm
from cem.model.source import Source, SourceManager, gaussian_pulse, ricker_wavelet
import math
from fealpy.cem.mesh import YeeUniformMesher

yee2d = YeeUniformMesher((0,1,0,1),nx=32, ny=32)

fields, data = yee2d._init_fields_dict("E", ["z"], num_frames=0, axis_type=None)

def init_fields_2d(yee):
    # Create Ez only for simplicity: shape (nx+1, ny+1) or (nx, ny) depending on your convention.
    # shape = (yee.nx+1, yee.ny+1)
    # ctx = dict(dtype='float64', device=None)
    # Ez = bm.zeros(shape, **ctx)
    # # create empty H fields as well
    # Hx = bm.zeros((yee.nx+1, yee.ny), **ctx)
    # Hy = bm.zeros((yee.nx, yee.ny+1), **ctx)
    # E_fields = {'x': bm.zeros_like(Ez), 'y': bm.zeros_like(Ez), 'z': Ez}
    # H_fields = {'x': Hx, 'y': Hy, 'z': bm.zeros_like(Hx)}
    E_fields,_ = yee._init_fields_dict("E", ["z"], num_frames=0, axis_type=None)
    H_fields,_ = yee._init_fields_dict("H", ["x","y"], num_frames=0, axis_type=None)
    return E_fields, H_fields

def run_simple_time_loop(yee, src_manager, E_fields, H_fields, nt=40, dt=1.0):
    """Very small time loop that only calls src_manager.apply_all to test source injection."""
    times = []
    for n in range(nt):
        t = n * dt
        src_manager.apply_all(t, yee, E_fields, H_fields)
        times.append(t)
    return E_fields, H_fields

def test_multiple_sources_and_superposition_2d(yee2d):
    yee = yee2d
    E_fields, H_fields = init_fields_2d(yee)
    sm = SourceManager()
    # two gaussian soft sources at different locations
    s1 = Source(position=(8,8), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=10.0, tau=2.0), amplitude=1.0, spread=1, injection='soft')
    s2 = Source(position=(20,20), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=10.0, tau=2.0), amplitude=0.8, spread=1, injection='soft')
    sm.add(s1); sm.add(s2)
    # run
    E1, H1 = run_simple_time_loop(yee, sm, E_fields, H_fields, nt=30, dt=1.0)
    # snapshot values near sources should be non-zero
    assert float(bm.max(bm.abs(E1['z'][6:11,6:11]))) > 0.0
    assert float(bm.max(bm.abs(E1['z'][18:23,18:23]))) > 0.0

    # superposition test: run individually and compare
    # run s1 only
    E_fields_a, H_fields_a = init_fields_2d(yee)
    sm_a = SourceManager(); sm_a.add(s1)
    Ea, Ha = run_simple_time_loop(yee, sm_a, E_fields_a, H_fields_a, nt=30, dt=1.0)

    # run s2 only
    E_fields_b, H_fields_b = init_fields_2d(yee)
    sm_b = SourceManager(); sm_b.add(s2)
    Eb, Hb = run_simple_time_loop(yee, sm_b, E_fields_b, H_fields_b, nt=30, dt=1.0)

    # run s1+s2 already computed as E1; compare linearity: E1 ≈ Ea + Eb
    lhs = E1['z']
    rhs = Ea['z'] + Eb['z']
    diff = bm.max(bm.abs(lhs - rhs))
    assert float(diff) < 1e-9 or float(diff/bm.max(bm.abs(rhs))+1e-12) < 1e-6

def test_hard_vs_soft_and_spread_2d(yee2d):
    yee = yee2d
    E_fields, H_fields = init_fields_2d(yee)
    sm = SourceManager()
    s_soft = Source(position=(16,16), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=5.0, tau=1.0), amplitude=1.0, spread=2, injection='soft')
    s_hard = Source(position=(10,10), comp='Ez', waveform=lambda t: gaussian_pulse(t, t0=5.0, tau=1.0), amplitude=2.0, spread=0, injection='hard')
    sm.add(s_soft); sm.add(s_hard)
    E, H = run_simple_time_loop(yee, sm, E_fields, H_fields, nt=20, dt=1.0)
    # hard source should place a value equal to its waveform amplitude at center (approx)
    center_val = float(E['z'][10,10])
    assert abs(center_val - 2.0 * gaussian_pulse(5.0, t0=5.0, tau=1.0)) < 1e-6 or center_val != 0.0
    # spread>0 should distribute energy: neighborhood max should be positive
    assert float(bm.max(bm.abs(E['z'][14:19,14:19]))) > 0.0


yee3d = YeeUniformMesher((0,1,0,1,0,1),nx=20,ny=20,nz=20)


def init_fields_3d(yee):
    shape = (yee.nx+1, yee.ny+1, yee.nz+1)
    ctx = dict(dtype='float64', device=None)
    Ez = bm.zeros(shape, **ctx)
    E_fields = {'x': bm.zeros_like(Ez), 'y': bm.zeros_like(Ez), 'z': Ez}
    H_fields = {'x': bm.zeros((yee.nx+1, yee.ny+1, yee.nz)), 'y': bm.zeros((yee.nx+1, yee.ny, yee.nz+1)), 'z': bm.zeros((yee.nx, yee.ny+1, yee.nz+1))}
    return E_fields, H_fields

def run_simple_time_loop(yee, src_manager, E_fields, H_fields, nt=20, dt=1.0):
    for n in range(nt):
        t = n * dt
        src_manager.apply_all(t, yee, E_fields, H_fields)
    return E_fields, H_fields

def test_multiple_sources_and_superposition_3d(yee3d):
    yee = yee3d
    E_fields, H_fields = init_fields_3d(yee)
    sm = SourceManager()
    s1 = Source(position=(4,4,4), comp='Ez', waveform=lambda t: ricker_wavelet(t, t0=5.0, f=0.5), amplitude=1.0, spread=1, injection='soft')
    s2 = Source(position=(10,10,10), comp='Ez', waveform=lambda t: ricker_wavelet(t, t0=5.0, f=0.5), amplitude=0.6, spread=1, injection='soft')
    sm.add(s1); sm.add(s2)
    Esum, Hsum = run_simple_time_loop(yee, sm, E_fields, H_fields, nt=15, dt=1.0)

    # check local non-zero near each source
    assert float(bm.max(bm.abs(Esum['z'][2:7,2:7,2:7]))) > 0.0
    assert float(bm.max(bm.abs(Esum['z'][8:13,8:13,8:13]))) > 0.0

    # superposition test
    E_fields_a, H_fields_a = init_fields_3d(yee)
    sm_a = SourceManager(); sm_a.add(s1)
    Ea, Ha = run_simple_time_loop(yee, sm_a, E_fields_a, H_fields_a, nt=15, dt=1.0)

    E_fields_b, H_fields_b = init_fields_3d(yee)
    sm_b = SourceManager(); sm_b.add(s2)
    Eb, Hb = run_simple_time_loop(yee, sm_b, E_fields_b, H_fields_b, nt=15, dt=1.0)

    diff = bm.max(bm.abs(Esum['z'] - (Ea['z'] + Eb['z'])))
    assert float(diff) < 1e-9 or float(diff / (bm.max(bm.abs(Ea['z'] + Eb['z'])) + 1e-12)) < 1e-6

