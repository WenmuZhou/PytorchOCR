import sys
import os
import subprocess

python_path = sys.executable

ori_path = os.getcwd()
os.chdir('torchocr/postprocess/pse_postprocess/pse')
if subprocess.call(
        '{} setup.py build_ext --inplace'.format(python_path), shell=True) != 0:
    raise RuntimeError(
        'Cannot compile pse: {}, if your system is windows, you need to install all the default components of `desktop development using C++` in visual studio 2019+'.
        format(os.path.dirname(os.path.realpath(__file__))))
os.chdir(ori_path)

from .pse import pse
