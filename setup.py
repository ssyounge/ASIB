from setuptools import setup, find_packages

setup(
    name="asib_kd",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[l.strip() for l in open("requirements.txt")],
    python_requires=">=3.8",
)
