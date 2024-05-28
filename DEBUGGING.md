# Debugging Information

## Binding New Methods
1. Make sure the method is exported in the Timeloop build.

E.g., use `nm -D libtimeloop-mapper.so` to inspect the library.

2. Make sure that the correct Timeloop library is being linked.
