import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SplittedImage",
    version="0.0.3",
    author="GoatWang",
    author_email="jeremywang@thinktronltd.com",
    description="For splitting satellite image",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://rd.thinktronltd.com/jeremywang/SplittedImage",
    packages=setuptools.find_packages(),
    # package_data={'PySaga': ['saga_cmd_pkls/*']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
#    install_requires=["Django >= 1.1.1",],
)

# 0.0.0
# 0.0.2 modify write_combined_tif to fit the image not well splitted by the right box size
# 0.0.3 add Normalizer

# python3 setup.py sdist bdist_wheel
# scp ~/Projects/PySaga/dist/PySaga-0.0.3-py3-none-any.whl  thinktron@rd.thinktronltd.com:/home/thinktron/pypi/PySaga-0.0.3-py3-none-any.whl
# pip3 install -U --index-url http://192.168.0.167:28181/simple --trusted-host 192.168.0.167 PySaga
