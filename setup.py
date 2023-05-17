from setuptools import setup, find_packages

# To make the project pip installable
setup(
    name="fmr",  # Frame Recognition
    version="1.0.0",
    description="A simple command line frame recognition program.",
    author="Erick Carrillo, Raudel Casas, Andres Peña",
    author_email="sansepiol26@gmail.com",
    url="https://github.com/RaudelCasas1603/Image-Classification",  # Update with the correct URL

    packages=find_packages(exclude=["test*"]),
    install_requires=[
        "opencv-python",
        "scikit-learn",
        "tqdm",
        "tensorflow",
        # Add other dependencies here
    ],
    
    # Add the features to here
    entry_points={
        "console_scripts": [
            "fmr = src.__main__:main",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    license = "MIT",
)
