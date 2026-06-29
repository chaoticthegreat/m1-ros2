import os
from glob import glob

from setuptools import find_packages, setup

package_name = "m1_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # MoveIt config (Phase 3): SRDF + kinematics/OMPL/controllers yaml for
        # planned, collision-aware moves (m1_moveit.launch.py reads these).
        (os.path.join("share", package_name, "moveit"),
            glob("moveit/*.srdf") + glob("moveit/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jerry",
    maintainer_email="jerry@example.com",
    description="Launch files and RViz config for the M1 ROS 2 control stack.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
