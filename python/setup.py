from setuptools import setup

setup(
  name = 'asforests',
  packages = ['asforests'],
  version = '0.0.1',
  license='MIT',
  description = 'Automatically Stopping Random Forests',
  author = 'Felix Mohr',                   # Type in your name
  author_email = 'mail@felixmohr.de',      # Type in your E-Mail
  url = 'https://github.com/fmohr/asforests',   # Provide either the link to your github or to your website
  keywords = ['random forests', 'early stopping', 'sklearn'],
  install_requires=[
          'numpy',
          'scikit-learn',
          'scipy',
          'tqdm'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ],
)
