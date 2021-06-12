from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

requirements = [
    "numpy", "scikit-learn", "scipy", "matplotlib", "nibabel", "brainsmash"]

setup(
    name="braintrips",
    version="0.0.0",
    author="Joshua Burt",
    author_email="joshuaburtphd@gmail.com",
    include_package_data=True,
    description="BRainwide Activity Induced by Neuromodulation via TRanscriptomics-Informed Pharmacological Simulation.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/murraylab/braintrips",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
