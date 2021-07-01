from setuptools import setup, find_packages

REQUIRED_PKGS = ['scikit-learn==0.22.1']
TESTS_REQUIRE = ['pytest==5.2.2']

setup(
    name='tinydp',
    version='0.0.1',
    author='Gabriele Modena',
    author_email='gmodena@pm.me',
    packages=find_packages(),
    license='Apache',
    install_requires=REQUIRED_PKGS,
    tests_require=TESTS_REQUIRE,
    long_description=open('README.md').read(),
)