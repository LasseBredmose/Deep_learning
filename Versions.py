import numpy
import sklearn
import matplotlib
import torch
#import tensorflow
import torchvision

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
print('The numpy version is {}.'.format(numpy.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The matplotlib version is {}.'.format(matplotlib.__version__))
print('The torch version is {}.'.format(torch.__version__))
#print('The tensorflow version is {}.'.format(tensorflow.__version__))
print('The torchvision version is {}.'.format(torchvision.__version__))

"""
module load python3/3.6.13 matplotlib/3.3.4-numpy-1.19.5-python-3.6.13 numpy/1.19.5-python-3.6.13-openblas-0.3.13     
"""
