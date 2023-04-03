from setuptools import setup, find_packages

setup(
    name='AIAF-Stroke',
    version='0.1.0',
    description='Predicting the diagnosis of arrhythmia in early stroke from clinical, brain imaging, and cardiac biology features.',
    authors=['Thibault Ellong','Hakim Saghir'] 
    author_email=['thibaultellong59@gmail.com','hakim.saghir@outlook.com']
    url='https://github.com/hakim-saghir/D4GEN',
    packages=find_packages(),

    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.16.0',
        'pandas>=0.24.0',
        'streamlit>=0.49.0',
        'matplotlib>=3.0.0',
        'xgboost>=0.90',
        'scikit-learn>=0.21.0',
        'scipy>=1.2.0'
    ],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Doctors',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)