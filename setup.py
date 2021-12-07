from setuptools import setup

setup(
    name='msslib',
    version='0.1.0',
    url='https://github.com/boothmanrylan/msslib/',
    description='A python port of Justin Braatens msslib: '
                'https://github.com/gee-community/msslib',
    author='Rylan Boothman',
    author_email='boothmanrylan@gmail.com',
    packages=['msslib'],
    install_requires=['earthengine-api', 'IPython', 'folium']
)
