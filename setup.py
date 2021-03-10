from setuptools import setup

setup(
    name='Main',
    version='1.0',
    description='Reconhecimento facial',
    author='paulo sergio',
    author_email='paulo_sergio2006@hotmail.com.br',
    packages=['Main'],
    install_requires=['Keras==2.4.3',
                      'joblib==0.17.0',
                      'mysql-connector-python==8.0.22',
                      'numpy==1.18.5',
                      'opencv-python==4.4.0.44',
                      'mysql-connector-python==8.0.22',
                      'scikit-learn==0.22.2.post1',
                      'tensorflow==2.3.0',
                      'tensorflow-cpu==2.3.0',
                      'tensorflow-estimator==2.3.0',
                      'tensorflow-gpu==2.3.0',
                      'tensorflow-gpu-estimator==2.3.0',
                      'Keras-Preprocessing==1.1.2',
                      'matplotlib==3.3.2',
                      'mtcnn==0.1.0'
                      ],

)
