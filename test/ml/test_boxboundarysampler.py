import pytest
from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import BoxBoundarySampler

from boxboundarysampler_data import bc_random, bc_linspace, bc_option



class TestBoxBoundarySampler:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("test_case", bc_random)
    def test_random_mode(self, backend, test_case):
        bm.set_backend(backend)
        domain = test_case['domain']
        m = test_case['m']
        ndim = len(domain) // 2
        sampler = BoxBoundarySampler(domain, mode='random', boundary=None, dtype=bm.float64)
        samples = sampler.run(m)
        assert samples.shape[1] == ndim
        
        for i in range(ndim):
            lower_bound = (samples[:, i] == domain[2*i])
            upper_bound = (samples[:, i] == domain[2*i+1])
            assert bm.any(lower_bound | upper_bound), f"维度{i}没有点在边界上"

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("test_case", bc_linspace)
    def test_linspace_mode(self, backend, test_case):
        bm.set_backend(backend)
        domain = test_case['domain']
        m = test_case['m']
        ndim = len(domain) // 2
        sampler = BoxBoundarySampler(domain, mode='linspace', boundary=None, dtype=bm.float64)
        samples = sampler.run(m)  # 包含边界点
        assert samples.shape[1] == ndim
        
        boundary_flags = bm.zeros(samples.shape[0], dtype=bool)
        for i in range(ndim):
            lower_bound = (samples[:, i] == domain[2*i])
            upper_bound = (samples[:, i] == domain[2*i+1])
            boundary_flags |= (lower_bound | upper_bound)
        assert bm.all(boundary_flags), "存在点不在任何边界上"

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("test_case", bc_option)
    @pytest.mark.parametrize("mode", ['random', 'linspace'])
    def test_selected_boundaries(self, backend, test_case, mode):
        bm.set_backend(backend)
        domain = test_case['domain']
        m = test_case['m']
        boundary = test_case['boundary']
        ndim = len(domain) // 2

        sampler = BoxBoundarySampler(domain, mode=mode, boundary=boundary, dtype=bm.float64)
        if mode == 'random':
            samples = sampler.run(m)
        else:
            samples = sampler.run(m)
        
        assert samples.shape[1] == ndim

        for b in boundary:
            dim = b // 2
            bound_value = domain[b] 
            assert bm.any(samples[:, dim] == bound_value), f"边界{b}采样点不符合要求"
            
            # 验证其他维度在范围内
            for other_dim in range(ndim):
                if other_dim != dim:
                    assert bm.all(samples[:, other_dim] >= domain[2*other_dim])
                    assert bm.all(samples[:, other_dim] <= domain[2*other_dim+1])
