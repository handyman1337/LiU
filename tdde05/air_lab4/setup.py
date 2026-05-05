from setuptools import find_packages, setup

package_name = 'air_lab4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arviv790',
    maintainer_email='arviv790@student.liu.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'visualise_semantic_objects = air_lab4.visualise_semantic_objects:main',
            'generate_tst = air_lab4.generate_tst:main',
        ],
    },
)
