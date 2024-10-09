import io

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

install_requires = [requirement for requirement in requirements if '--' not in requirement]

setup(
    name='depthai-sdk',
    version='1.15.1',
    description='This package provides an abstraction of the DepthAI API library.',
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/luxonis/depthai/tree/main/depthai_sdk',
    keywords="depthai sdk oak camera",
    author='Luxonis',
    author_email='support@luxonis.com',
    license='MIT',
    packages=['depthai_sdk', 'depthai_sdk_console_scripts'],
    package_dir={'': 'src'},  # https://stackoverflow.com/a/67238346/5494277
    install_requires=install_requires,
    include_package_data=True,
    extras_require={
        "visualize": ['PySide2',
                      'Qt.py>=1.3.0',
                      'matplotlib==3.5.3; python_version <= "3.7"',
                      'matplotlib==3.6.1; python_version > "3.7"',
                      'depthai-viewer',
                      'open3d'
                      ],
        "replay": ['mcap>=0.0.10',
                   'mcap-ros1-support==0.0.8',
                   'rosbags==0.9.11'],
        "record": ['av'],
        "test": ['pytest']
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
    entry_points={
        'console_scripts': [
            'depthai_sdk = depthai_sdk_console_scripts.depthai_sdk:main',
        ]
    },
)
