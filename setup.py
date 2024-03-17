from setuptools import find_packages,setup
from typing import List
 
HYPHEN_E_DOT='-e .'

def get_requirements(filepath:str)->List[str]:
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements =[requirement.replace('\n','') for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements
setup(
    name='Diamond Price Predictor',
    version='0.0.1',
    author='Anirudh',
    author_email='anirudh7371@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)