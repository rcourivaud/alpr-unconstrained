from setuptools import find_packages
from distutils.core import setup, Extension

with open('requirements.txt') as f:
	requirements = f.read().splitlines()

module = Extension('demo',
                    define_macros=[('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0')],
                    include_dirs=['license_plate_extractor/darknet'],
                    library_dirs=['license_plate_extractor/darknet'],
                    sources=['license_plate_extractor/darknet/src/demo.c'])

setup(
	name='license_plate_extractor',
	version='0.1',
	packages=find_packages(),
	url='',
	license='',
	install_requires=requirements,
	include_package_data=True,
	zip_safe=False,
	author='Raphael Courivaud',
	author_email='r.courivaud@gmail.com',
	description='',
	ext_modules = [module]
)
