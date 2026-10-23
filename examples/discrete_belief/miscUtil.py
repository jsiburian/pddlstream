

class SymbolGenerator:
    """
    Generate new symbols guaranteed to be different from one another
    Optionally, supply a prefix for mnemonic purposes
    Call gensym("foo") to get a symbol like 'foo37'
    """

    def __init__(self):
        self.count = 0

    def gensym(self, prefix='i', zeroPadded=False):
        self.count += 1
        if zeroPadded:
            return (prefix + '_%08i') % self.count
        else:
            return prefix + '_' + str(self.count)


gensym = SymbolGenerator().gensym
"""Call this function to get a new symbol"""


# Return a flat list of elements.  Stuff should be a list.
# Turn a list of sets into a single set using union
# Return a dictionary
# stuff is a list of dictionaries, where the values are sets.
#  merge so
# that, in the result, each key is associated with the union of the
# sets that were their values in the individual dictionaries.
# Squash one level
# If equiv is true, don't map two objects that should be different
# into the same string

# if eq = True, use 6 digits otherwise 3

def clip(v, vMin, vMax):
    """
    @param v: number
    @param vMin: number (may be None, if no limit)
    @param vMax: number greater than C{vMin} (may be None, if no limit)
    @returns: If C{vMin <= v <= vMax}, then return C{v}; if C{v <
    vMin} return C{vMin}; else return C{vMax}
    """
    if vMin is None:
        if vMax is None:
            return v
        else:
            return min(v, vMax)
    else:
        if vMax is None:
            return max(v, vMin)
        else:
            return max(min(v, vMax), vMin)


def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """

    return max(((x, f(x)) for x in l), key=lambda x: x[1])[0]


def argmin(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the lowest score
    """

    return min(((x, f(x)) for x in l), key=lambda x: x[1])[0]


def argmaxWithVal(l, f):
    vals = [f(x) for x in l]
    maxV = max(vals)
    return l[vals.index(maxV)], maxV


def update(x, **entries):
    # Update a dict; or an object with slots; according to entries.
    # >>> update({'a': 1}, a=10, b=20)
    # {'a': 10, 'b': 20}
    # >>> update(Struct(a=1), a=10, b=20)
    # Struct(a=10, b=20)

    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x


# Doesn't extend bindings if there are bindings in s
def applyBindings1(s, bindings):
    if not bindings:
        return s
    tps = type(s)
    if tps == str:
        if s in bindings:
            return applyBindings1(bindings[s], bindings)
        else:
            return s
    if tps == dict:
        return dict(applyBindings1(s.items(), bindings))
    elif tps == tuple:
        return tuple([applyBindings1(thing, bindings) for thing in s])
    elif tps == set:
        return set([applyBindings1(thing, bindings) for thing in s])
    elif tps == list:
        return [applyBindings1(thing, bindings) for thing in s]
    elif hasattr(s, 'applyBindings'):
        return s.applyBindings(bindings)
    else:
        return s


applyBindings = applyBindings1


# A string starting with an upper case letter is a planning operator
# schema variable
'''
# Assumes funTable is monotonic
def inverseTableLookup(pQuery, table):
    for i in range(len(table)):
        (r, p) = funTable[i]
        if pQuery < p:
            if i == 0: return r
            # interpolate
            (rp, pp) = table[i-1]
            # dr / dp
            rate = (r - rp) / (p - pp)
            return pp + (pQuery - pp) * rate
    return table[-1][p]
'''


# Variable name manipulation
# names are of the form: pred(obj)_numbers

# Bindings == None means we've failed
# if renaming is True, then only let variables match variables
# Returns bindings, no side effect.  One-sided wrt '*': that is a
# constant in t1 will match a * in t2, but not vv.
# Find variable renaming only
# Returns True or False
# Find variable renaming only
# Returns True or False
# Return the largest element of dom that is less than v.
# in two dimensions
# Returns list of lists
