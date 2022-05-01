from grpc import composite_channel_credentials
import setuptools

PYTHON_REQUIRES = ">=3.7"
INSTALL_REQUIRES = [
    "scikit-learn = 1.0.2",
    "plotly = ^5.5.0",
    "tqdm = ^4.62.3",
    "python-box = ^5.4.1",
    "ttach = ^0.0.3",
    "category-encoders = ^2.3.0",
    "lightgbm = ^3.3.2",
    "xgboost = ^1.5.2",
]



setuptools.setup(
   name= "yusuke_library",
   author="yusuke orito",
   description="my library for competition",
   version="0.1.0",
   url="https://github.com/yusukeorito/my_lib.git",
   python_requires=PYTHON_REQUIRES,

)