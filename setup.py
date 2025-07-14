# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Python setup script."""

import argparse
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    args.git_version = None
    args.long_description = ""
    if args.version is None and os.path.exists("version.txt"):
        with open("version.txt", "r") as f:
            args.version = f.read().strip()
    if os.path.exists(".git"):
        try:
            git_version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="./")
            args.git_version = git_version.decode("ascii").strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists("README.md"):
        with open(os.path.join("README.md"), encoding="utf-8") as f:
            args.long_description = f.read()
    return args


def clean_builds():
    for path in ["build", "diffnext.egg-info"]:
        if os.path.exists(path):
            shutil.rmtree(path)


def find_packages(top):
    """Return the python sources installed to package."""
    packages = []
    for root, _, _ in os.walk(top):
        if os.path.exists(os.path.join(root, "__init__.py")):
            packages.append(root)
    return packages


def find_package_data():
    """Return the external data installed to package."""
    return []


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        with open("diffnext/version.py", "w") as f:
            f.write(
                'version = "{}"\n'
                'git_version = "{}"\n'
                "__version__ = version\n".format(args.version, args.git_version)
            )
        super(BuildPyCommand, self).build_packages()

    def build_package_data(self):
        self.package_data = {"diffnext": find_package_data()}
        super(BuildPyCommand, self).build_package_data()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    def initialize_options(self):
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


args = parse_args()
setuptools.setup(
    name="diffnext",
    version=args.version,
    description="A diffusers based library for autoregressive diffusion models.",
    long_description=args.long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baaivision/NOVA",
    author="BAAI",
    license="Apache License",
    packages=find_packages("diffnext"),
    cmdclass={"build_py": BuildPyCommand, "install": InstallCommand},
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "imageio-ffmpeg",
        "omegaconf",
        "wandb",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
clean_builds()
