from setuptools import setup, find_packages

setup(
  name = 'anymal-belief-state-encoder-decoder-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.17',
  license='MIT',
  description = 'Anymal Belief-state Encoder Decoder - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/anymal-belief-state-encoder-decoder-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention gating',
    'belief state',
    'robotics'
  ],
  install_requires=[
    'einops>=0.4',
    'einops-exts',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
