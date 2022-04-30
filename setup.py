from grpc import composite_channel_credentials
import setuptools

PYTHON_REQUIRES = ">=3.7"


setuptools.setup(
   name= "yusuke_library",
   author="yusuke orito",
   description="my library for competition",
   version="0.1.0",
   url="https://github.com/yusukeorito/my_lib.git",
   python_requires=PYTHON_REQUIRES,

)