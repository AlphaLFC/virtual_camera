from setuptools import setup, find_packages

setup(
    name='virtual_camera',
    version='0.0.4',
    description='Virtual Camera',
    author='AlphaLFC',
    author_email='alphali@motovis.com',
    url='https://github.com/AlphaLFC/virtual_camera',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
        '': ['data/*']
    },
    install_requires=[
        'numpy',
        'scipy'
    ]
)
