import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="upstride",
    version="2.0",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="A package to use Geometrical Algebra in pure tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://upstride.io",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    # test_suite="",
    # tests_require=[''],
    # install_requires=[
    #     '',
    # ],
)