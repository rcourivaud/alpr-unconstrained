from setuptools import find_packages
from distutils.core import setup, Extension

with open('requirements.txt') as f:
	requirements = f.read().splitlines()

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
)
