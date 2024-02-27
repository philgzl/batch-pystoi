from setuptools import find_packages, setup

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='batch_pystoi',
    version='0.0.2',
    description=('Short Term Objective Intelligibility metric for batched '
                 'inputs'),
    author='Manuel Pariente, Philippe Gonzalez',
    author_email='pariente.mnl@gmail.com, hello@philgzl.com',
    url='https://github.com/philgzl/batch-pystoi',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
