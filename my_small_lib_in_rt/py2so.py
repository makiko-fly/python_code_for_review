import argparse
import os

# setup.py
from distutils.core import setup
from Cython.Build import cythonize
import glob

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--py_name', dest='py_name', default=None, help='py file name')
    args = parser.parse_args()

    if not args.py_name:
        raise Exception('please specify python name to convert via -n or --py_name option')

    py_path = os.path.join(CUR_DIR, args.py_name + '.py')
    print(py_path)
    if not os.path.exists(py_path):
        raise Exception('py file at {} not found'.format(py_path))

    setup(ext_modules=cythonize(glob.glob(py_path)),
          script_args=['build_ext', '--inplace'])

if __name__ == '__main__':
    main()
