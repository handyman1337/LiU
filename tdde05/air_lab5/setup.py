from setuptools import find_packages, setup

package_name = 'air_lab5'

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
    maintainer='linja937',
    maintainer_email='linja937@student.liu.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'text_to_goals_node = air_lab5.text_to_goals_node:main',
            'decision_node = air_lab5.decision_node:main'
        ],
    },
)
