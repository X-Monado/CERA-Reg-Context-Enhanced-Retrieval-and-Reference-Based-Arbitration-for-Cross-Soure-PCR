from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='cera-reg',
    packages=find_packages(),
    description='Context-enhanced retrieval and reference-based arbitration for point cloud registration',
    version='0.1.0',
    ext_modules=[
        # These sources expose CPU grid subsampling / radius search only.
        # A CPU extension avoids requiring nvcc/CUDA_HOME just to rebuild the
        # Python/PyTorch ABI for a configured training environment.
        CppExtension(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extensions/extra/cloud/cloud.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/extensions/pybind.cpp',
            ],
            extra_compile_args=['-O2', '-g0'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
