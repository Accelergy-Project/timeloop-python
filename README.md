# timeloop-python Python wrapper for the timeloop project.

## Dependencies
Since building PyTimeloop requires building Timeloop, dependencies of
Timeloop are also required.
```
// Timeloop dependencies
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
libyaml-cpp-dev
libncurses-dev
libtinfo-dev
libgpm-dev

// PyTimeloop dependencies
cmake
```

## Installing
Recrusively clone the repo
```
 git clone --recurse-submodules git@github.com:Accelergy-Project/timeloop-python.git
```

Update the git submodules using
```
$ git submodule update
```
Point to your Timeloop source and built libraries:
```
export TIMELOOP_INCLUDE_PATH=your/timeloop/path/include
export TIMELOOP_LIB_PATH=directory/with/timeloop/libs
```
Note that Timeloop needs to be built to generate dynamic libraries.

Then, install PyTimeloop by running
```
$ pip3 install -e .
```
If you ran `pip3 install -e .` recently, the `build/` directory has to be
cleaned by running `rm -rf build`.


## Using Command Line Tools
After installing PyTimeloop, there are some premade Timeloop applications you
can use on the command line:
- `timeloop-model.py`

For example,
```
$ timeloop-model.py --help
usage: timeloop-model.py [-h] [--output_dir OUTPUT_DIR]
                         [--verbosity VERBOSITY]
                         configs [configs ...]

Run Timeloop given architecture, workload, and mapping.

positional arguments:
  configs               Config files to run Timeloop.

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory to dump output.
  --verbosity VERBOSITY
                        0 is only error; 1 adds warning; 2 is everyting.
```

## Contributing
This README is written with users as its audience, more information relevant
to the development of the project can be found in CONTRIBUTING.md.

## General Debugging Information
1. When debugging the C++ bindings, it may be faster to build by calling `cmake`
   directly instead of `pip3 install`

## Known Bugs
1. segmentation fault at the end of timeloop-mapper.py. Doesn't break anything,
   but quite annoying. Might be caused when cleaning up some objects.
