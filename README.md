# timeloop-python
Python wrapper for the timeloop project.

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
Update the git submodules using
```
git submodule update
```
Timeloop has to be built manually using
```
scons -j4 --accelergy
```
Then, install PyTimeloop by running
```
pip3 install -e .
```
If you ran `pip3 install -e .` recently, the `build/` directory has to be
cleaned. For example, by running `rm -rf build`.

### Using your own build of Timeloop
If you want to use your own build of Timeloop, you can set the environment
variable `LIBTIMELOOP_PATH` to your Timeloop directory that contains the
Timeloop binary (`libtimeloop-model.so`).

For example, if you have the library in `/path/to/library/libtimeloop-model.so`
you can execute the following
```
export LIBTIMELOOP_PATH=/path/to/library
rm -rf build && pip3 install -e .
```
