# ==================== setup.py ====================
from setuptools import setup, find_packages

setup(
    name="crackdetect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "pillow",
        "scikit-image",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
    author="Paul",
    author_email="paulosdahprogrammer@gmail.com",
    description="A tool for analyzing construction images for cracks",
)
