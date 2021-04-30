from collections.abc import Mapping, Iterable
from abc import ABC, abstractmethod
from typing import Union, IO

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from bindings import NativeConfig


class Config(ABC):
    def __new__(cls):
        return ConfigDict()

    @classmethod
    def load_yaml(cls, stream: Union[bytes, IO[bytes], str, IO[str]]):
        loaded_yaml = yaml.load(stream, Loader=Loader)
        if isinstance(loaded_yaml, dict):
            return ConfigDict(loaded_yaml)
        else:
            raise ValueError('Top-level config file is always dict-like.')

    @abstractmethod
    def dump_yaml(self):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError()

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @abstractmethod
    def __sizeof__(self):
        raise NotImplementedError()

    def get_native(self):
        self.native_config = NativeConfig()
        self.native_config.load_yaml(self.root.dump_yaml())
        self.native_config_node = self.native_config.get_root()
        for key in self.root_key:
            self.native_config_node = self.native_config_node[key]
        return self.native_config, self.native_config_node

    @staticmethod
    def is_scalar(x):
        return (isinstance(x, bool) or isinstance(x, str) or isinstance(x, int)
                or isinstance(x, float))


class ConfigDict(Config):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, d: dict = None, root: Config = None, root_key=[]):
        if not d:
            d = {}
        self.dict = d

        if not root:
            root = self
        self.root = root
        self.root_key = root_key

        if root == self:
            # Set constraint to default no constraint
            self['architecture_constraints'] = {}
            self.canonicalize_names()

        # Set up Timeloop native wrappers
        self.native_config = None
        self.native_config_node = None
        self.get_native()

    def dump_yaml(self):
        return yaml.safe_dump(dict(self.dict))

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, key):
        res = self.dict[key]
        child_root_key = self.root_key + [key]
        if isinstance(res, dict):
            return ConfigDict(res, root=self.root, root_key=child_root_key)
        elif isinstance(res, list):
            return ConfigList(res, root=self.root, root_key=child_root_key)
        elif isinstance(res, set):
            raise ValueError('Set is not supported')
        else:
            return res

    def __contains__(self, item):
        return item in self.dict

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return ('ConfigDict(d=%s, root=%s, root_key=%s'
                % (self.dict, self.root, self.root_key))

    def __sizeof__(self):
        return len(self.dict)

    def canonicalize_names(self):
        if 'arch' in self:
            self['architecture'] = self['arch']
        if 'architecture' in self and 'constraints' in self['architecture']:
            self['architecture_constraints'] = \
                self['architecture']['constraints']
        if 'arch_constraints' in self:
            self['architecture_constraints'] = self['arch_constraints']


class ConfigList(Config):
    def __new__(cls, l: list = None, root: Config = None, root_key=[]):
        return object.__new__(cls)

    def __init__(self, l: list = None, root: Config = None, root_key=[]):
        if not l:
            l = []
        self.lis = l

        if not root:
            root = self
        self.root = root
        self.root_key = root_key
        # Set up Timeloop native wrappers
        self.native_config = None
        self.native_config_node = None
        self.get_native()

    def dump_yaml(self):
        return yaml.safe_dump(list(self.lis))

    def __setitem__(self, idx, value):
        self.lis[idx] = value

    def __getitem__(self, idx):
        res = self.lis[idx]
        child_root_key = self.root_key + [idx]
        if isinstance(res, dict):
            return ConfigDict(res, root=self.root, root_key=child_root_key)
        elif isinstance(res, list):
            return ConfigList(res, root=self.root, root_key=child_root_key)
        elif isinstance(res, set):
            raise ValueError('Set is not supported')
        else:
            return res

    def __contains__(self, item):
        return item in self.lis

    def __str__(self):
        return str(self.lis)

    def __repr__(self):
        return ('ConfigList(l=%s, root=%s, root_key=%s'
                % (self.lis, self.root, self.root_key))

    def __sizeof__(self):
        return len(self.lis)
