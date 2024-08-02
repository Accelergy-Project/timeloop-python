# About
The PyTimeloop project is a combination of Python bindings to the Timeloop
library as well as TimeloopFE. The Timeloop Front-End (TimeloopFE) is a Python
front-end interface to the Timeloop infrastructure, which allows users to model
tensor accelerators and explore the vast space of architectures, workloads, and
mappings.

TimeloopFE provides a rich Python interface, error checking, and automation
tools. With closely-aligned Python and YAML interfaces, TimeloopFE is designed
to enable easy design space exploration and automation.

## Documentation
Documentation for the full framework is available at
[timeloop.csail.mit.edu](https://timeloop.csail.mit.edu). Documentation for TimeloopFE
is available at
[accelergy-project.github.io/timeloopfe/index.html](https://accelergy-project.github.io/timeloopfe/index.html).

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

## Installation
First, ensure that Timeloop and Accelergy are installed following the
[Timeloop+Accelergy install instructions](https://timeloop.csail.mit.edu/installation).

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

## Tutorials and Examples
Tutorials and examples available in the [Timeloop and Accelergy exercises
repository](https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git).
In this repository, examples can be found in the `workspace/baseline_designs`
directory and tutorials can be found in the `workspace/exercises` directory.

## Minimal Usage
TimeloopFE interface provides two primary functions: - Input file gathering &
error checking - Python interface for design space exploration
```python 
import pytimeloop.timeloopfe.v4 as tl
from joblib import Parallel, delayed

# Basic setup. Gathers input files, checks for errors
spec = tl.Specification.from_yaml_files(
  "your_input_file.yaml", "your_other_input_file.yaml"
)
# Call Timeloop mapper
tl.call_mapper(spec, output_dir="your_output_dir")
# Call Accelergy verbose
tl.call_accelergy_verbose(spec, output_dir="your_output_dir")

# Multiprocessed design space exploration
def run_mapper_with_spec(buf_size: int):
  spec = tl.Specification.from_yaml_files(
    "your_input_file.yaml", "your_other_input_file.yaml"
  )
  spec.architecture.find("my_buffer").attributes.depth = buf_size
  return tl.call_mapper(spec, output_dir=f"outputs_bufsize={buf_size}")

buf_sizes = [1024, 2048, 4096, 8192, 16384]
results = Parallel(n_jobs=8)(
  delayed(run_mapper_with_spec)(buf_size) for buf_size in buf_sizes
)
```

Please visit the [Timeloop and Accelergy exercises
repository](https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git)
for more examples and tutorials.

## General Debugging Information
1. When debugging the C++ bindings, it may be faster to build by calling `cmake`
   directly instead of `pip3 install`

## Citation
Please cite the following:

- A. Parashar, P. Raina, Y. S. Shao, Y.-H. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. Emer, “Timeloop: A systematic approach to DNN accelerator evaluation,” in 2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2019, pp. 304–315.
- M. Horeni, P. Taheri, P. Tsai, A. Parashar, J. Emer, and S. Joshi, “Ruby: Improving hardware efficiency for tensor algebra accelerators through imperfect factorization,” in 2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2022, pp. 254–266.
- Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, “Sparseloop: An analytical, energy-focused design space exploration methodology for sparse tensor accelerators,” in 2021 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2021, pp. 232–234.
- Y. N. Wu, J. S. Emer, and V. Sze, “Accelergy: An architecture-level energy estimation methodology for accelerator designs,” in 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2019, pp. 1–8.
- T. Andrulis, J. S. Emer, and V. Sze, “CiMLoop: A flexible, accurate, and fast compute-in-memory modeling tool,” in 2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2024.

Or use the following BibTeX:

```BibTeX
@inproceedings{timeloop,
  author      = {Parashar, Angshuman and Raina, Priyanka and Shao, Yakun Sophia and  Chen, Yu-Hsin and Ying, Victor A and Mukkara, Anurag and Venkatesan, Rangharajan and Khailany, Brucek and Keckler, Stephen W and Emer, Joel},
  booktitle   = {2019 IEEE international symposium on performance analysis of systems and software (ISPASS)}, pages={304--315}, year={2019},
  title       = {Timeloop: A systematic approach to dnn accelerator evaluation},
  year        = {2019},
}
@inproceedings{ruby,
  author      = {M. Horeni and P. Taheri and P. Tsai and A. Parashar and J. Emer and S. Joshi},
  booktitle   = {2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  title       = {Ruby: Improving Hardware Efficiency for Tensor Algebra Accelerators Through Imperfect Factorization},
  year        = {2022},
}
@inproceedings{sparseloop,
  author      = {Wu, Yannan N. and Tsai, Po-An, and Parashar, Angshuman and Sze, Vivienne and Emer, Joel S.},
  booktitle   = {{ ACM/IEEE International Symposium on Microarchitecture (MICRO)}},
  title       = {{Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling }},
  year        = {{2022}}
}
@inproceedings{accelergy,
  author      = {Wu, Yannan Nellie and Emer, Joel S and Sze, Vivienne},
  booktitle   = {2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  title       = {Accelergy: An architecture-level energy estimation methodology for accelerator designs},
  year        = {2019},
}
@inproceedings{cimloop,
  author      = {Andrulis, Tanner and Emer, Joel S. and Sze, Vivienne},
  booktitle   = {2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)}, 
  title       = {{CiMLoop}: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool}, 
  year        = {2024},
}
```
