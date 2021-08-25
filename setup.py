import io

from setuptools import setup

setup(
    name='depthai-sdk',
    version='0.0.1',
    description='This package contains convenience classes and functions that help in most common tasks while using DepthAI API',
    long_description=io.open("depthai_sdk/README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/luxonis/depthai/sdk',
    keywords="depthai sdk utils managers previews helpers",
    author='Luxonis',
    author_email='support@luxonis.com',
    license='MIT',
    packages=['sdk'],
    install_requires=[
        "requests",
        "PyYAML",
        "boto3"
    ],
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/luxonis/depthai/issues",
        "Source Code": "https://github.com/luxonis/depthai/tree/main/sdk",
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)