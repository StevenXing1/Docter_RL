from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="final_proj",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Reinforcement Learning Doctor Agent for Hypotension Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Docter_RL",
    packages=find_packages(exclude=["tests", "scripts", "logs", "models"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "numpydoc>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "doctor-train=scripts.train:main",
            "doctor-eval=scripts.evaluate:main",
            "doctor-viz=scripts.visualize:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
