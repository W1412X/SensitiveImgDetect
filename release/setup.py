from setuptools import setup, find_packages

setup(
    name='SensitiveImgDetect',
    version='0.1.5',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'pillow',
        'torchvision'
    ],
    author='W1412X',
    author_email='w108418@126.com',
    description='A package used to detect sensitive image',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/W1412X/SensitiveImgDetect',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
