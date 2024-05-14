from setuptools import setup, find_packages

setup(
    name='virtual_camera',
    version='0.0.3',
    description='Virtual Camera',
    author='AlphaLFC',
    author_email='',
    long_description=open('virtual_camera/README.md').read(),
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
