import os
from glob import glob

from setuptools import find_packages, setup

package_name = "m1_can_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    # The served config page ships inside the package so the node finds it via
    # __file__ even in a plain (non-symlink) install.
    package_data={package_name: ["web/*.html"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "web"), glob("m1_can_tools/web/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jerry",
    maintainer_email="jerry@example.com",
    description="M1 Damiao CAN bring-up driver + hardware config/test web page.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "m1_hwconfig = m1_can_tools.hwconfig_node:main",
        ],
    },
)
