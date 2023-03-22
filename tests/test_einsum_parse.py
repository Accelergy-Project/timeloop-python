import unittest
from pytimeloop.problem import Einsum

class TestEinsumParse(unittest.TestCase):
    def test_conv1d_without_coefs(self):
        e = Einsum.from_yaml("""
            problem:
                shape:
                    name: Conv1D_OC
                    dimensions: [ K, R, P ]
                    data-spaces:
                    - name: Weights
                      projection:
                      - [ [K] ]
                      - [ [R] ]
                    - name: Inputs
                      projection:
                      - [ [R], [P] ]
                    - name: Outputs
                      projection:
                      - [ [K] ]
                      - [ [P] ]
                    read-write: True

                instance:
                    K: 32
                    R: 3
                    P: 16
        """)

    def test_conv1d_with_coefs_default(self):
        e = Einsum.from_yaml("""
            problem:
              shape:
                name: Conv1D_OC
                dimensions: [ K, R, P ]
                data-spaces:
                - name: Weights
                  projection:
                  - [ [K] ]
                  - [ [R] ]
                - name: Inputs
                  projection:
                  - [ [Dilation, R], [Stride, P] ]
                - name: Outputs
                  projection:
                  - [ [K] ]
                  - [ [P] ]
                read-write: True
                coefficients:
                - default: 1
                  name: Stride
                - default: 1
                  name: Dilation

              instance:
                K: 32
                R: 3
                P: 16
        """)

    def test_conv1d_with_coefs_override(self):
        e = Einsum.from_yaml("""
            problem:
              shape:
                name: Conv1D_OC
                dimensions: [ K, R, P ]
                data-spaces:
                - name: Weights
                  projection:
                  - [ [K] ]
                  - [ [R] ]
                - name: Inputs
                  projection:
                  - [ [Dilation, R], [Stride, P] ]
                - name: Outputs
                  projection:
                  - [ [K] ]
                  - [ [P] ]
                read-write: True
                coefficients:
                - default: 1
                  name: Stride
                - default: 1
                  name: Dilation

              instance:
                K: 32
                R: 3
                P: 16
                Stride: 3
                Dilation: 2
        """)