"""Objects: the values that streams take and produce (Object), and placeholders for values that have not been
sampled yet (OptimisticObject)."""

from collections import namedtuple
from itertools import count

from pddlstream.core.registry import get_registry
from pddlstream.core.syntax import get_parameter_name
from pddlstream.core.utils import CreationHash, is_hashable, str_from_object

USE_HASH = True
USE_OBJ_STR = True
USE_OPT_STR = True
OPT_PREFIX = '#'
PREFIX_LEN = 1


class Object(CreationHash):
    """A value that appears in facts, named v0, v1, ... (or by a constant's name). Objects live in the current
    registry (core/registry.py): the same value always maps to the same Object within it."""

    _prefix = 'v'

    def __init__(self, value, stream_instance=None, name=None):
        registry = get_registry()
        self.value = value
        self.index = len(registry.obj_from_name)
        if name is None:
            name = f'{self._prefix}{self.index}'
        self.pddl = name
        self.stream_instance = stream_instance  # TODO: store first created stream instance
        registry.obj_from_id[id(self.value)] = self
        registry.obj_from_name[self.pddl] = self
        if is_hashable(value):
            registry.obj_from_value[self.value] = self

    def is_unique(self):
        return True

    def is_shared(self):
        return False

    @staticmethod
    def from_id(value):
        obj_from_id = get_registry().obj_from_id
        if id(value) not in obj_from_id:
            return Object(value)
        return obj_from_id[id(value)]

    @staticmethod
    def has_value(value):
        registry = get_registry()
        if USE_HASH and not is_hashable(value):
            return id(value) in registry.obj_from_id
        return value in registry.obj_from_value

    @staticmethod
    def from_value(value):
        if USE_HASH and not is_hashable(value):
            return Object.from_id(value)
        obj_from_value = get_registry().obj_from_value
        if value not in obj_from_value:
            return Object(value)
        return obj_from_value[value]

    @staticmethod
    def has_name(name):
        return name in get_registry().obj_from_name

    @staticmethod
    def from_name(name):
        return get_registry().obj_from_name[name]

    @staticmethod
    def reset():
        registry = get_registry()
        registry.obj_from_id.clear()
        registry.obj_from_value.clear()
        registry.obj_from_name.clear()

    def __lt__(self, other):  # For heapq on python3
        return self.index < other.index

    def __repr__(self):
        if USE_OBJ_STR:
            return str_from_object(self.value)  # str
        return self.pddl


##################################################


class UniqueOptValue(namedtuple('UniqueOptTuple', ['instance', 'sequence_index', 'output'])):
    @property
    def parameter(self):
        return self.output


class SharedOptValue(namedtuple('SharedOptTuple', ['stream', 'inputs', 'input_objects', 'output'])):
    @property
    def values(self):
        return tuple(obj.value for obj in self.input_objects)


##################################################


class DebugValue:  # TODO: could just do an object
    _prefix = '@'  # $ | @

    def __init__(self, stream, input_values, output_parameter):
        self.stream = stream
        self.input_values = input_values
        self.output_parameter = output_parameter
        self.index = next(get_registry().output_counts[type(self)][output_parameter])  # Counted per class

    def __repr__(self):
        # Can also just return first letter of the prefix
        return f'{self._prefix}{get_parameter_name(self.output_parameter)}{self.index}'


class SharedDebugValue(namedtuple('SharedDebugValue', ['stream', 'output_parameter'])):
    # TODO: this alone doesn't refining at the shared object level
    _prefix = '&'  # $ | @ | &

    def __repr__(self):
        return f'{self._prefix}{get_parameter_name(self.output_parameter)}'


##################################################

# TODO: just one object class or have Optimistic extend Object
# TODO: make a parameter class that has access to some underlying value


class OptimisticObject(CreationHash):
    """A placeholder for a stream output that planning assumes exists (#o0, or e.g. #q1 for an output ?q), kept in
    the current registry."""

    _prefix = f'{OPT_PREFIX}o'  # $ % #

    def __init__(self, value, param):
        # TODO: store first created instance
        registry = get_registry()
        self.value = value
        self.param = param
        self.index = len(registry.opt_from_inputs)
        if USE_OPT_STR and isinstance(self.param, UniqueOptValue):
            # TODO: instead just endow UniqueOptValue with a string function
            parameter = self.param.output
            prefix = get_parameter_name(parameter)[:PREFIX_LEN]
            var_index = next(registry.opt_counts.setdefault(prefix, count()))
            self.repr_name = f'{OPT_PREFIX}{prefix}{var_index}'
            self.pddl = self.repr_name
        else:
            self.pddl = f'{self._prefix}{self.index}'
            self.repr_name = self.pddl
        registry.opt_from_inputs[(value, param)] = self
        registry.opt_from_name[self.pddl] = self

    def is_unique(self):
        return isinstance(self.param, UniqueOptValue)

    def is_shared(self):
        return not isinstance(self.param, UniqueOptValue)  # OptValue

    @staticmethod
    def from_opt(value, param):
        # TODO: make param have a default value?
        key = (value, param)
        opt_from_inputs = get_registry().opt_from_inputs
        if key not in opt_from_inputs:
            return OptimisticObject(value, param)
        return opt_from_inputs[key]

    @staticmethod
    def has_name(name):
        return name in get_registry().opt_from_name

    @staticmethod
    def from_name(name):
        return get_registry().opt_from_name[name]

    @staticmethod
    def reset():
        registry = get_registry()
        registry.opt_from_inputs.clear()
        registry.opt_from_name.clear()
        registry.opt_counts.clear()

    def __lt__(self, other):  # For heapq on python3
        return self.index < other.index

    def __repr__(self):
        return self.repr_name
