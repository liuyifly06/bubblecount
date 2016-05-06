try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Bubble counting for imaging analysis of SDD',
    'author': 'Yi Liu',
    'url': 'https://github.com/liuyifly06/bubblecount.git',
    'download_url': 'git clone https://github.com/liuyifly06/bubblecount.git',
    'author_email': 'liuyifly@gmail.com',
    'version': '0.1',
    'install_requires': ['scikit-image','tensorflow'],
    'packages': ['bubblecount'],
    'scripts': [],
    'name': 'bubblecount'
}

setup(**config)
