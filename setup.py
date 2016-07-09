from setuptools import setup, find_packages


setup(
    name='anomaly_detection',
    version='0.0.1a',
    author='Thiago Curvelo',
    author_email='tcurvelo@gmail.com',
    description='Document Anomaly Detection',
    packages=find_packages(),
    include_package_data=True,
    install_requires = [
        'nltk',
        'numpy',
        'setuptools',
        'unidecode',
    ],
)
