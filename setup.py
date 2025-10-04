"""
Setup script for RayBand Voice Camera.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rayband-voice-cam",
    version="1.0.0",
    author="RayBand Team",
    author_email="team@rayband.dev",
    description="Voice-controlled camera system for smart glasses",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SoloScriptSage/rayband-voice-cam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "rayband=rayband.cli.main:main",
            "rayband-test=scripts.test_hardware:main",
            "rayband-setup=scripts.setup_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rayband": [
            "hardware/components/*",
        ],
    },
    zip_safe=False,
)
