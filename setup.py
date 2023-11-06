from distutils.core import setup

setup(
    name='SpykeTools',
    version='1.0',
    description='Fast, unsupervised discovery of high-dimensional neural spiking patterns based on optimal transport theory',
    author='Boris Sotomayor-Gomez',
    author_email='bsotomayor92@gmail.com',
    py_modules=[
        'spyketools.clustering', 
        'spyketools.distances', 
        'spyketools.io', 
        'spyketools.manifold', 
        'spyketools.utils', 
        'spyketools.vis',
        ],
)
