from setuptools import setup, find_packages

setup(
  name = 'anymal-belief-state-encoder-decoder-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.21',
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
    'assoc-scan>=0.0.2',
    'einops>=0.8',
    'evolutionary-policy-optimization>=0.0.61',
    'torch>=2.2',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
