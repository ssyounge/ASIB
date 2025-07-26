from setuptools import setup, find_packages

setup(
    name="asib_kd",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        ln.strip()
        for ln in open("requirements.txt")
        if ln.strip() and not ln.startswith("#")
    ],
    python_requires=">=3.8",
)
