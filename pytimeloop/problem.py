import bindings
import yaml
from yaml.loader import FullLoader


class Workload(bindings.problem.Workload):
    def __init__(self, config):
        super().__init__(config)

class Einsum:
    """
    TODO: this should be integrated with workload somehow.
    """
    def __init__(self, id):
        self.id = id
        self.name = ''
        self.out_proj: str = ''
        self.out_dspace: str = ''
        self.in_dspace_to_proj: dict[str, str] = {}

    @staticmethod
    def from_yaml(f, id=0):
        einsum_yaml = yaml.load(f, Loader=FullLoader)

        if 'problem' not in einsum_yaml:
            raise ValueError('Misshapen einsum file')
        problem = einsum_yaml['problem']

        if 'shape' not in problem or 'instance' not in problem:
            raise ValueError('Misshapen einsum file')
        shape = einsum_yaml['problem']['shape']
        instance = einsum_yaml['problem']['instance']

        dims = shape['dimensions']

        coefs = {}
        if 'coefficients' in shape:
            for c in shape['coefficients']:
                coefs[c['name']] = c['default']

        dspaces = shape['data-spaces']

        einsum = Einsum(id)
        einsum.name = shape['name']

        for dspace in dspaces:
            name = dspace['name']
            proj = dspace['projection']
            isl_proj = Einsum._isl_aff_from_projection(proj, dims, coefs,
                                                       instance)
            if 'read-write' in dspace and dspace['read-write']:
                einsum.out_dspace = name
                einsum.out_proj = isl_proj
            else:
                einsum.in_dspace_to_proj[name] = isl_proj

    @staticmethod
    def _isl_aff_from_projection(proj, dims, coefs, instance):
        make_factor_str = lambda factor: \
            Einsum._make_factor_str(
                factor,
                dims,
                coefs,
                instance
            )

        proj_str_lis = []
        for dim_proj in proj:
            terms_str_lis = []
            for term in dim_proj:
                factors_str = map(make_factor_str, term)
                terms_str_lis.append('*'.join(factors_str))
            proj_str_lis.append('+'.join(terms_str_lis))
        
        projs_str = ','.join(proj_str_lis)
        dims_str = ','.join(dims)
        isl_aff = f'{{ [{dims_str}] -> [{projs_str}] }}'
        return isl_aff

    @staticmethod
    def _make_factor_str(factor, dims, coefs, instance):
        if factor in dims:
            return factor
        if factor in instance:
            return str(instance[factor])
        if factor in coefs:
            return str(coefs[factor])
        raise ValueError(f'Unknown factor {factor} in a projection')
