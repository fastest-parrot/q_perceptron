import setuptools

with open("version.txt", "r") as v:
    version = v.read()

setuptools.setup(
    name='q_perceptron',
    author='Sean Kennedy',
    description='Simple 1-layer perceptron built in qutip',
    version=version,
    install_requires=[
        'qtip', 'sklearn', 'pandas', 'matplotlib', 'seaborn', 'numpy'
    ],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    #package_data={'q_perceptron': ['data']},
    #include_package_data=True
)
