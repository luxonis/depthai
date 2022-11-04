import io

from setuptools import setup

with open('requirements.txt') as f:
    required = f.readlines()

setup(
    name='depthai-sdk',
    version='1.9.0',
    description='This package provides an abstraction of the DepthAI API library.',
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/luxonis/depthai/tree/main/depthai_sdk',
    keywords="depthai sdk oak camera",
    author='Luxonis',
    author_email='support@luxonis.com',
    license='MIT',
    packages=['depthai_sdk'],
    package_dir={"": "src"},  # https://stackoverflow.com/a/67238346/5494277
    install_requires=required,
    include_package_data=True,
    extras_require={
        "visualize": ["PySide2", "Qt.py>=1.3.0"]
    },
    project_urls={
        "Bug Tracker": "https://github.com/luxonis/depthai/issues",
        "Source Code": "https://github.com/luxonis/depthai/tree/main/depthai_sdk",
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
