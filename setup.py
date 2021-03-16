import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['Neural-Painter']
#from version import __version__

setup(
  name = 'Neural-Painter',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'Neural Painters: A learned differentiable constraint for generating brushstroke paintings',
  author = 'Shauray Singh',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/Neural-Painter',
  keywords = ['generative adversarial networks', 'machine learning','VAE'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
  ],
)
