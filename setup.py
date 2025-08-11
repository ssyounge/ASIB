from pathlib import Path
from setuptools import setup, find_packages

req_path = Path(__file__).resolve().parent / "requirements.txt"
with req_path.open(encoding='utf-8') as f:
    install_requires = [
        ln.strip()
        for ln in f
        if ln.strip() and not ln.startswith(("#", "-e", "--"))
    ]

setup(
    name="asib_kd",
    version="0.2.0",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)
