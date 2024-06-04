from setuptools import setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open("README.md", "r", encoding="utf-8") as fp:
    readme = fp.read()

setup(
    name="easy_llm",
    version="1.0.0",
    description="This package is designed to make it easy to run various LLMs.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="k141303",
    author_email="kouta.nakayama@gmail.com",
    maintaner="k141303",
    maintaner_email="kouta.nakayama@gmail.com",
    packages=["easy_llm"],
    package_dir={"": "src"},
    url="https://github.com/k141303/easy_llm",
    download_url="https://github.com/k141303/easy_llm",
    include_package_data=True,
    install_requires=_requires_from_file("requirements.txt"),
)
