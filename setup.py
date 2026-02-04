from setuptools import setup, find_packages

setup(
    name="modiff",
    packages=find_packages(),
    version="0.1.0",
    description="Motion Diffusion",
    author="Chaitanya Patel",
    include_package_data=True,
    package_data={"": ["py.typed"]},
)
