from setuptools import setup, Extension, Command, find_packages
from setuptools.command.build_py import build_py as build_py_orig

from distutils.sysconfig import get_config_vars as default_get_config_vars
import distutils.sysconfig as dsc

from Cython.Build import cythonize
from Cython.Compiler import Options

import glob, sys, os

#remove python doc strings in the so files 
Options.docstrings = False
Options.emit_code_comments = False
# Additional flags used to compile python files
compile_args = ["-fvisibility=protected",
                "-ffast-math", 
                "-funroll-loops", 
                "-floop-nest-optimize", 
                "-fipa-pta", 
                "-flto", 
                "-ftree-vectorize", 
                "-fno-signed-zeros", 
                "-fvect-cost-model=unlimited"]
# By default we aiming the architectures on where we are compiling
arch = 'native'
if '--arch' in sys.argv:
    index = sys.argv.index('--arch')
    sys.argv.pop(index)  # Removes the '--arch'
    arch = sys.argv.pop(index)  # Returns the element after the '--arch'
# The arch is now ready to use for the setup

# Arch optimization
# Specify the name of the target processor, optionally suffixed by one or more feature modifiers. 
# This option has the form -mcpu=cpu{+[no]feature}* /-march=arch{+[no]feature}*
if arch == "aarch64":
    # The values 'cortex-a57' specify that GCC should tune for a big.LITTLE system, with crypto and simd optimizations
    # Recommanded for jetson by nvidia: https://forums.developer.nvidia.com/t/gcc-options-for-tx2/56669
    compile_args.extend(["-march=armv8-a+crypto+simd","-mcpu=cortex-a57+crypto+simd"])
elif arch == "x86_64":
    compile_args.extend(["-march=skylake","-mavx2","-mfma"])
else:
    compile_args.append(f'-march={arch}')

# Add all python files recursively to "Extension"
# Exclude all __init__.py and test*.py
py_files = [f for f in glob.glob('src/python/upstride/**/*.py',recursive=True) if not ("__" in f or "test" in f)]
ext_names = [x.split('.')[0].replace('/','.') for x in py_files]
ext_modules_list = list()
for name, pyfile in zip(ext_names, py_files):
    ext_modules_list.append( Extension(name,
                                      [pyfile],
                                      libraries=[],
                                      library_dirs=[],
                                      language="c++", # Compile for c++
                                      extra_compile_args=compile_args)) 

# This is required so that .py files are not added to the build folder
class build_py(build_py_orig):
    def build_packages(self):
        pass

# Remove all debug flag that can add not required information
def flags_modification(x):
    if type(x) is str:
        x = x.strip()
        x = x.replace("-DNDEBUG", "")
        x = x.replace("-g "," ")
        x = x.replace("-c "," ")
        x = x.replace("-D_FORTIFY_SOURCE=2","-D_FORTIFY_SOURCE=1")
        x = x.replace("-O1","-O3")
        x = x.replace("-O2","-O3")
        x = x.replace("-fstack-protector-strong","")
    if type(x) is list:
        for sub_x in x:
            flags_modification(sub_x)
    elif type(x) is dict:
        for _,sub_x in x.items():
            flags_modification(sub_x)
    return x


def modify_config_vars(*args):
  result = default_get_config_vars(*args)
  # sometimes result is a list and sometimes a dict:
  if type(result) is list:
     return [flags_modification(x) for x in result]
  elif type(result) is dict:
     return {k : flags_modification(x) for k,x in result.items()}
  else:
     raise Exception("cannot handle type"+type(result))

# replace original get_config_vars to the updated one.    
dsc.get_config_vars = modify_config_vars

# define a cleanup function
class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./*.egg-info ./build/cython')


setup(
    name="upstride",
    version="2.0",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="UpStride Engine package.",
    long_description="A package to use UpStride technology to improve Tensorflow.",
    long_description_content_type="text/markdown",
    url="https://upstride.io",
    include_package_data=True,
    packages=['upstride'],
    package_dir={'upstride': 'src/python/upstride/type_generic'},
    package_data={'upstride': ['libupstride.so', 'libdnnl.so.1']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    # language_level=3 for python3 
    ext_modules= cythonize(ext_modules_list, language_level=3, nthreads=4, build_dir="build/cython"),
    cmdclass = {
        'build_py': build_py,
        'clean': CleanCommand
    }
)

