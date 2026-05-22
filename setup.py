from setuptools import setup, find_packages

setup(
    name="simple_driving",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "simple_driving.resources": ["*.urdf"],
    },
    install_requires=[
        "gymnasium",
        "pybullet",
        "numpy",
    ],
)
