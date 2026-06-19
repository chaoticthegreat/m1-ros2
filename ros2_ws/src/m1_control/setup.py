import os
from glob import glob

from setuptools import find_packages, setup

package_name = "m1_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "web"), glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jerry",
    maintainer_email="jerry@example.com",
    description="Whole-body controller for the M1 robot (DLS arm/lift reach + swerve base).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "m1_controller = m1_control.controller_node:main",
            "m1_send_pose = m1_control.send_pose:main",
            "m1_teleop = m1_control.teleop_node:main",
            "m1_web = m1_control.web_node:main",
            "m1_quest = m1_control.quest_node:main",
        ],
    },
)
