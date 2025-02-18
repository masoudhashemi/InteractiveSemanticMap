from setuptools import find_packages, setup

setup(
    name="semantic_landscape",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "sentence-transformers",
        "minisom",
        "litellm>=1.16.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.8.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="An interactive semantic landscape system using SOMs and LLMs",
    keywords="nlp, som, clustering, machine-learning",
    url="https://github.com/yourusername/semantic-landscape",
)
