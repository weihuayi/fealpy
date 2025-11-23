import pytest

from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import ISampler

from isampler_data import inter_linspace, inter_random

class TestISampler:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("test_case", inter_random)
    def test_random_mode(self, backend, test_case):
        domain = test_case['domain']
        m = test_case['m']
        bm.set_backend(backend)
        sampler = ISampler(domain, mode='random', dtype=bm.float64)
        samples = sampler.run(m)
        
        assert samples.shape[0] == m
        assert samples.shape[1] == len(domain) // 2
        for i in range(len(domain) // 2):
            lower = domain[2*i]
            upper = domain[2*i+1]
            assert bm.all(samples[:, i] > lower)
            assert bm.all(samples[:, i] < upper)
        if backend == 'pytorch':
            assert samples.requires_grad == False
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("test_case", inter_linspace)
    def test_linspace_mode(self, backend, test_case):
        bm.set_backend(backend)
        domain = test_case['domain']
        sampler = ISampler(domain, mode='linspace', dtype=bm.float64, requires_grad=True)
        
        if isinstance(test_case['m'], int):
            samples = sampler.run(test_case['m'], rm_ends=True)
            expected_points = test_case['m'] ** (len(domain) // 2)
        else:
            samples = sampler.run(*test_case['m'], rm_ends=True)
            expected_points = 1
            for m in test_case['m']:
                expected_points *= m
                
        # 验证样本形状
        assert samples.shape[0] == expected_points
        assert samples.shape[1] == len(domain) // 2
        
        # 验证每个点是否在区域内
        for i in range(len(domain) // 2):
            lower = domain[2*i]
            upper = domain[2*i+1]
            assert bm.all(samples[:, i] > lower)
            assert bm.all(samples[:, i] < upper)
        
        if backend == 'pytorch':
            assert samples.requires_grad == True
        
