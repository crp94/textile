import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='textile',
     version='0.0.1',
     author="Carlos Rodriguez-Pardo",
     author_email="carlos.rodriguezpardo.jimenez@gmail.com",
     description="TexTile: A Differentiable Metric for Texture Tileability",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/crp94/textile",
     packages=['textile-metric'],
     install_requires=["torch>=1.2.0", "torchvision>=0.17.1", "einops>=0.7.0", "numpy>=1.14.3", "opencv-python>=2.4.11", "kornia>=0.7.2", "progressbar>=2.5"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
         "Development Status :: 4 - Beta",
         "License :: OSI Approved :: MIT License",
         "Topic :: Scientific/Engineering :: Image Processing",
         "Topic :: Scientific/Engineering :: Artificial Intelligence",
     ],
 )