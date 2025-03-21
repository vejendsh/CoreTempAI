from setuptools import setup, find_packages

setup(
    name="CoreTempAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    author="Sai Yeshwanth Vejendla",
    author_email="vejendsh@mail.uc.edu",
    description="Physics-based Machine Learning for Temperature Prediction",
    keywords="cfd, temperature, prediction, neural network, neural operator",
    python_requires=">=3.8", 
) 