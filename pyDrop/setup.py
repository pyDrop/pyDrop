import setuptools

with open("README.md", "r") as handle:
    long_description = handle.read()

setuptools.setup(
    name="pyDrop",

    author="pyDrop LLC",
    author_email="pydrop@colorado.edu",

    description=long_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",

    keywords="clustering, ddPCR, coarsegraining",

    packages=setuptools.find_packages(),
    include_package_data=True,

    classifiers=[
        "Development Status :: 1 - Alpha",
    ],
    entry_points={'console_scripts': []},
)