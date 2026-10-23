from .miscUtil import argmaxWithVal

import random
import copy


# Think of this as a mixture of a distribution and something like a uniform,
# but it doesn't really work unless the universe is finite
class Distribution:

    def prob(self, x):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def sample(self):
        return self.draw()


class DiscreteDist(Distribution):
    """
    Probability distribution over a discrete domain.  This is a
    generic superclass;  do not instantiate it.  Any subclass has to
    provide methods: C{prob}, C{setProb}, C{support}
    """

    def setProb(self, x, p):
        raise NotImplementedError()

    def support(self):
        raise NotImplementedError()

    def draw(self):
        """
        @returns: a randomly drawn element from the distribution
        """
        # TODO: numpy.random.choice
        r = random.random()
        total = 0.0
        for v in self.support():
            total += self.prob(v)
            if r < total:
                return v
        raise Exception('Failed to draw from:' + str(self))

    def expectation(self, vals):
        return sum(self.prob(x) * vals[x] for x in self.support())

    def expectationF(self, f):
        return sum(self.prob(x) * f(x) for x in self.support())

    def maxProbElt(self):
        """
        @returns: The element in this domain with maximum probability
        """
        return argmaxWithVal(self.support(), self.prob)

    # Returns the x with highest weight
    def mode(self):
        return self.maxProbElt()[0]

    # Estimate the mean.  Assumes that + and * are defined on the elements
    def mean(self):
        return sum(x * self.prob(x) for x in self.support())

    # Estimate the variance.  Assumes we can do x * x, and then scale
    # and sum the results.
    def variance(self):
        mu = self.mean()
        return sum((x - mu) ** 2 * self.prob(x) for x in self.support())

    def conditionOnVar(self, index, value):
        """
        @param index: index of a variable in the joint distribution
        @param value: value of that variable

        @returns: new distribution, conditioned on variable C{i}
        having value C{value}, and with variable C{i} removed from all
        of the elements (it's redundant at this point).
        """
        newElements = [elt for elt in self.support() if elt[index] == value]
        z = sum(self.prob(elt) for elt in newElements)
        return DDist({removeElt(elt, index): self.prob(elt) / z
                      for elt in newElements})

    def verify(self, epsilon=1e-5):
        z = sum(self.prob(elt) for elt in self.support())
        assert (1 - epsilon) < z < (1 + epsilon), 'degenerate distribution ' + str(self)


class DDist(DiscreteDist):
    """Discrete distribution represented as a dictionary.  Can be
    sparse, in the sense that elements that are not explicitly
    contained in the dictionary are assuemd to have zero probability."""

    def __init__(self, dictionary={}, name=None):
        self.__d = copy.copy(dictionary)
        """ Dictionary whose keys are elements of the domain and values
        are their probabilities. """
        self.name = name
        """Optional name;  used in sum-out operations"""
        self.normalize()

    def copy(self):
        return DDist(self.__d, self.name)

    def update(self):
        pass

    def computeMPE(self):
        # returns pair: (element, prob)
        return max(self.__d.items(), key=lambda pair: pair[1])

    maxProbElt = computeMPE

    def mode(self):
        # just the element
        element, prob = self.maxProbElt()
        return element

    def addProb(self, val, p):
        """
        Increase the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) + p)

    def mulProb(self, val, p):
        """
        Multiply the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) * p)

    def prob(self, elt):
        """
        @returns: the probability associated with C{elt}
        """
        return self.__d.get(elt, 0.0)

    def setProb(self, elt, p):
        """
        @param elt: element of the domain
        @param p: probability
        Sets probability of C{elt} to be C{p}
        """
        self.__d[elt] = p

    def support(self):
        """
        @returns: A list (in any order) of the elements of this
        distribution with non-zero probability.
        """
        return [x for x in self.__d.keys() if self.__d[x] > 0.0]

    def normalize(self, z=None):
        """
        Divides all probabilities through by the sum of the values to
        ensure the distribution is normalized.

        Changes the distribution!!  (And returns it, for good measure)

        Generates an error if the sum of the current probability
        values is zero.
        """
        if not z:
            z = sum(self.prob(e) for e in self.support())
        assert z > 0.0, 'degenerate distribution ' + str(self)
        alpha = 1.0 / z
        newD = {}
        for e in self.support():
            newD[e] = self.__d[e] * alpha
        self.__d = newD
        return self

    def normalizeOrSmooth(self):
        """
        If total prob is < 1, then spread that mass out uniformly
        (among elements with non-zero probabilty, seince we don't
        really know the universe; otherwiese, just normalize.
        """
        z = sum(self.prob(e) for e in self.support())
        assert z > 0.0, 'degenerate distribution ' + str(self)
        newD = DDist({})
        if z < 1:
            beta = (1 - z) / len(self.support())
            for e in self.support():
                newD.addProb(e, beta)
        return self.normalize()

    def smooth(self, totalP):
        # redistribute p of the probability mass to the values that have
        # positive support
        n = len(self.support())
        p = totalP / n
        for e in self.support():
            self.addProb(e, p)
        self.normalize()

    def blur(self, blurDomain, blurFactor):
        """
        Adds small prob to all elements in C{blurDomain} and
        renormalizes.  Side effects the distribution.
        @param blurDomain: set of elements to be blurred
        @param blurFactor: how much to blur; 1 obliterates everything,
                           0 has no effect
        """
        eps = blurFactor * (1.0 / len(blurDomain))
        for elt in blurDomain:
            self.addProb(elt, eps)
        self.normalize()

    def condition(self, test):
        return DDist({e: self.prob(e) for e in self.support() if test(e)})

    # Used to be called map
    def project(self, f):
        """
        Return a new distribution with the same probabilities, but
        with each element replaced by f(elt)
        """
        newD = {}
        for key, val in self.__d.items():
            newK = f(key)
            newD[newK] = val + newD.get(newK, 0.0)
        return DDist(newD)

    def transitionUpdate(self, tDist):
        new = {}
        for sPrime in self.support():  # for each old state sPrime
            tDistS = tDist(sPrime)  # distribution over ending states
            if tDistS is None:  # terminal state
                continue
            oldP = self.prob(sPrime)
            for s in tDistS.support():
                # prob of transitioning to s from sPrime
                new[s] = new.get(s, 0.0) + tDistS.prob(s) * oldP
        self.__d = new

    def obsUpdate(self, obs_model, obs):
        for si in self.support():
            self.mulProb(si, obs_model(si).prob(obs))
        z = sum(self.prob(e) for e in self.support())
        if 0.0 < z:
            self.normalize(z)
        return z

    def obsUpdates(self, obs_models, observations):
        # Avoids explicitly computing the joint distribution
        assert len(obs_models) == len(observations)
        for si in self.support():
            combo = to_tuple(si)
            for obs_model, obs in zip(obs_models, observations):
                self.mulProb(si, obs_model(*combo).prob(obs))
                combo = combo + (obs,)
        z = sum(self.prob(e) for e in self.support())
        if 0.0 < z:
            self.normalize(z)
        return z

    def __repr__(self):
        return '{}{{{}}}'.format(self.__class__.__name__, ', '.join(
            '{}: {:.5f}'.format(*pair) for pair in sorted(
                self.__d.items(), key=lambda pair: pair[1])))

    def __hash__(self):
        return hash(frozenset(self.__d.items()))

    def __eq__(self, other):
        return self.__d == other.__d

    def __ne__(self, other):
        return self.__d != other.__d


######################################################################
#   Special cases

def DeltaDist(v):
    """
    Distribution with all of its probability mass on value C{v}
    """
    return DDist({v: 1.0})


def UniformDist(elts):
    """
    Uniform distribution over a given finite set of C{elts}
    @param elts: list of any kind of item
    """
    p = 1.0 / len(elts)
    return DDist({e: p for e in elts})


class MixtureDist(DiscreteDist):
    """
    A mixture of two probability distributions, d1 and d2, with
    mixture parameter p.  Probability of an
    element x under this distribution is p * d1(x) + (1 - p) * d2(x).
    It is as if we first flip a probability-p coin to decide which
    distribution to draw from, and then choose from the appropriate
    distribution.

    This implementation is lazy;  it stores the component
    distributions.  Alternatively, we could assume that d1 and d2 are
    DDists and compute a new DDist.
    """

    def __init__(self, d1, d2, p):
        self.d1 = d1
        self.d2 = d2
        self.p = p
        self.binom = DDist({True: p, False: 1 - p})

    def prob(self, elt):
        return self.p * self.d1.prob(elt) + (1 - self.p) * self.d2.prob(elt)

    def draw(self):
        if self.binom.draw():
            return self.d1.draw()
        else:
            return self.d2.draw()

    def support(self):
        return list(set(self.d1.support()) | set(self.d2.support()))

    def makeDDist(self):
        return DDist({e: self.prob(e) for e in self.support()})

    def __str__(self):
        result = 'MixtureDist({'
        elts = self.support()
        for x in elts[:-1]:
            result += str(x) + ' : ' + str(self.prob(x)) + ', '
        result += str(elts[-1]) + ' : ' + str(self.prob(elts[-1])) + '})'
        return result

    __repr__ = __str__


def mixDDists(dists):
    # dists: a {dist: weight} dict or a sequence of (dist, weight) pairs.
    # Pairs keep equal components (e.g. two identical DeltaDists) from collapsing into one dict key.
    pairs = list(dists.items()) if isinstance(dists, dict) else list(dists)
    support = {key for dist, _ in pairs for key in dist.support()}
    assert support
    return DDist({key: sum(weight*dist.prob(key) for dist, weight in pairs)
                  for key in support})


def MixtureDD(d1, d2, p):
    """
    A mixture of two probability distributions, d1 and d2, with
    mixture parameter p.  Probability of an
    element x under this distribution is p * d1(x) + (1 - p) * d2(x).
    It is as if we first flip a probability-p coin to decide which
    distribution to draw from, and then choose from the appropriate
    distribution.

    This implementation is eager: it computes a DDist
    distributions.  Alternatively, we could assume that d1 and d2 are
    DDists and compute a new DDist.
    """
    return mixDDists([(d1, p), (d2, 1 - p)])


def JDist(PA, *PBgAs, **kwargs):
    """
    Create a joint distribution on P(A, B) (in that order),
    represented as a C{DDist}
        
    @param PA: a C{DDist} on some random var A
    @param PBgA: a conditional probability distribution specifying
    P(B | A) (that is, a function from elements of A to C{DDist}
    on B)
    """
    joint = PA.project(to_tuple)
    for PBgA in PBgAs:
        d = {}
        for a in joint.support():
            PB = PBgA(*a)
            for b in PB.support():
                combo = a + (b,)
                d[combo] = joint.prob(a) * PB.prob(b)
        joint = DDist(d, **kwargs)
    return joint


def totalProbability(PA, PBgA):
    PB = JDist(PA, PBgA)
    return PB.project(lambda pair: pair[1])


######################################################################
#   Continuous distribution

# Uses numpy matrices
# pose4 is a big hack to deal with a case when the last element is a rotation
# A mixture of Gaussians, with implicit "leftover" probability assigned to
# a uniform
# Rows of data are examples; mu is a column vector
######################################################################
# Factored distributions

######################################################################
# Multinomial distribution.  k items, probability p, independent, that
# each one will flip.  How many flips?

######################################################################
#   Utilities

def to_tuple(e):
    if isinstance(e, tuple):
        return e
    return (e,)


def removeElt(items, i):
    """
    non-destructively remove the element at index i from a list;
    returns a copy;  if the result is a list of length 1, just return
    the element  
    """
    result = items[:i] + items[i + 1:]
    if len(result) == 1:
        return result[0]
    else:
        return result


######################################################################
# Regression

# If we want the 1-pnm(delta) after an observation with obsSigma to be
# < epsr, then what does the 1-pnm have to be before the update?

# Amount of probability mass within delta of mean, given
# Amount of probability mass within delta of value, given a Gaussian
# Gauss CDF
# Given p value find chiSq

def clip(v, vMin, vMax):
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


### All much too specific to 2D.  Fix.

# Assume H  is identity (transforms state into obs)
