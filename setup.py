import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="depthai",
    version="0.0.3",
    author="Luxonis",
    author_email="brandon@luxonis.com",
    description="DepthAI Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luxonis/depthai",
    py_modules=["depthai"],
    packages=[''],
    package_data={'': ['*.so', '*.cmd']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
