import pathlib
from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="tarware",
    version="1.0.0",
    description="Task-Assignment Multi-Robot Warehouse environment for reinforcement learning",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Raul Steleac",
    url="https://github.com/raulsteleac/task-assignment-robotic-warehouse",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "numpy",
        "gym==0.21",
        "six==1.16.0",
        "pyglet==1.5.11",
        "networkx",
        "pyastar2d"
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
