from setuptools import setup, find_packages

def parse_requirements(fn):
    with open(fn) as f:
        return [req for req in f.read().rstrip().split('\n') if "==" in req and "#" not in req]


parsed_requirements = parse_requirements(
    'requirements.txt',
)

requirements = [str(ir) for ir in parsed_requirements]

with open('README.md') as description_file:
    description = description_file.read()

setup(
    name='aspectnlp',
    version='0.0.2',
    description="Aspect detection NLP toolkit is a Python package that perform NLP tasks based on aspect detection.",
    long_description=description,
    long_description_content_type='text/markdown',
    author="Shuanglu Dai",
    author_email='shuanglu.dai@gmail.com',
    packages=find_packages(include=['aspectnlp', 'aspectnlp.*']),
    package_dir={
        'aspectnlp': 'aspectnlp'
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='aspectnlp',
    py_modules=['mydatasets','misc'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "License :: OSI Approved :: MIT License",
    ]
)

