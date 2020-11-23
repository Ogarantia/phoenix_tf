from typing import List, Tuple
import functools
import numpy
import unittest


class CliffordProduct:
  """ Implements Clifford product.
  Callable entity performing the Clifford product of two multivectors.
  Intended to be used for testing purposes mainly.
  """
  def __init__(self, geometrical_def=(3, 0, 0), blade_ids=["", "1", "2", "3", "12", "13", "23", "123"]):
    """ Initializes Clifford product object
    Parameters:
        :geometrical_def:    metric signature
        :blade_idx:          blade indices
    """
    # blade_ids is a fancy list of strings; convert to a list of tuples for an easier digestion in Python
    self.blade_ids = [
        tuple([int(i) for i in entry]) for entry in blade_ids
    ]
    
    # specify algebra
    self.geometrical_def = geometrical_def
    self.dim = len(self.blade_ids)

    # compute blade squares according to the algebra signature
    self.squares = [0]
    possible_squares = [1, -1, 0]
    for i in range(3):
      self.squares += [possible_squares[i]] * self.geometrical_def[i]

    # make inverse mapping of blade_ids entries to their indices in blade_ids
    blade_ids_inverse = dict(zip(self.blade_ids, range(len(self.blade_ids))))

    # cache Clifford product matrices
    self.prod_signs = numpy.zeros((self.dim, self.dim), dtype=numpy.int32)      # sign resulting from multiplication of two blades
    self.prod_blades = numpy.zeros((self.dim, self.dim), dtype=numpy.int32)     # blade index resulting from multiplication of two blades
    for l in range(len(self.blade_ids)):
        for r in range(len(self.blade_ids)):
            s, i = self._multiply_blades(self.blade_ids[l], self.blade_ids[r])
            self.prod_signs[l, r] = s
            self.prod_blades[l, r] = blade_ids_inverse[i]

  def _multiply_blades(self, l1: Tuple, l2: Tuple) -> Tuple:
    """ Given e_{l1}, e_{l2} return (s, l) such as e_{l1} * e_{l2} = s * e_{l}
    l1, l2 and l are tuples of integers containing integer indices
    """
    # as l1 and l2 are already sorted, we can just merge them and count the number of permutation needed
    s = 1
    i1, i2, length_l1 = 0, 0, len(l1)
    out = []
    while i1 < len(l1) and i2 < len(l2):
      if l1[i1] == l2[i2]:
        # move the element of l2 near the element of l1 and remove them
        if (length_l1 - 1) % 2 != 0:
          s *= -1
        # check the sign of the square
        s *= self.squares[l1[i1]]
        length_l1 -= 1
        i1 += 1
        i2 += 1
      elif l1[i1] > l2[i2]:
        # then put the element of l2 in front of the element of l1
        if length_l1 % 2 != 0:
          s *= -1
        out.append(l2[i2])
        i2 += 1
      elif l1[i1] < l2[i2]:
        out.append(l1[i1])
        length_l1 -= 1
        i1 += 1
    out += l1[i1:] + l2[i2:]
    return s, tuple(out)

  def apply(self, op, inverse=False):
    """ Applies a multiplicative binary operation in the Clifford product sense.
    If op(i,j) = a[i] * b[j], computes Clifford product.
    Parameters:
        :op: the operation to apply accepting integer indices as arguments
        :inverse: FIXME
    """
    output = [None] * self.dim
    for i in range(self.dim):
      for j in range(self.dim):
        if not inverse:
          k, s = self.prod_blades[i, j], self.prod_signs[i, j]
        else:
          k, s = self.prod_blades[j, i], self.prod_signs[i, j]
        if s == 1:
          if output[k] is None:
            output[k] = op(i, j)
          else:
            output[k] += op(i, j)
        elif s == -1:
          if output[k] is None:
            output[k] = -op(i, j)
          else:
            output[k] -= op(i, j)
    return output

  def __call__(self, lhs, rhs):
    """ Computes Clifford product 
    Parameters:
        :lhs: a multivector, left operand of the product
        :rhs: a multivector, right operand of the product
    """
    assert len(lhs) == self.dim, "Number of dimensions does not match in left operand"
    assert len(rhs) == self.dim, "Number of dimensions does not match in right operand"
    return self.apply(lambda i, j: lhs[i] * rhs[j])

  def render_signtable(self):
    """ Computes the "signtable": description of Clifford product used internally in the engine.
    Finds the permutation of terms for the backpropagation.
    """
    # write out (left index, right index, positive) triples for all terms participating in the product
    positive_terms = []
    term_ctr = 0
    for d in range(self.dim):
      pos = self.prod_blades == d
      signs = self.prod_signs[pos]
      x, y = numpy.where(pos)
      print("//", d)
      for i in range(len(signs)):
        print("{ %d, %d, %s }," % (x[i], y[i], "true" if signs[i] > 0 else "false"))
        if signs[i] > 0:
          positive_terms.append((x[i], y[i], term_ctr))
        term_ctr += 1

    # find backpropagation order
    def recurse(l, r, order):
      """ Searches for a sequence of N terms satisfying the backprop order condition (N = algebra dimensionality):
      "For an N-dim algebra, first N terms need to contribute positively and cover every left and right component."
      It is assumed such a sequence always exists.
      :l:     left components included in the order discovered so far
      :r:     right components included in the order discovered so far
      :order: the order discovered so far.
      Returns true once the sequence stored in order satisfies the condition of having every left and right component
      with positive contribution
      """
      if len(l) == self.dim and len(r) == self.dim:
        # sequence found, print out (including the maining terms)
        remaining_terms = [i for i in range(self.dim ** 2) if i not in order]
        strs = [str(i) for i in order + remaining_terms]
        print(", ".join(strs))
        return True
      # some components are not covered. Run them through positive terms
      for term in positive_terms:
        # find the one containing left and right parts not yet included
        if term[0] not in l and term[1] not in r:
          # recursively process the remaining part
          if recurse(l + [term[0]], r + [term[1]], order + [term[2]]):
            return True
      return False
    # call
    recurse([], [], [])


class TestCliffordProduct(unittest.TestCase):
  def test_real(self):
    """ Basic sanity check of CliffordProduct implementing the real algebra
    """
    prod = CliffordProduct((0, 0, 0), [""])
    # real-valued algebra is one-dimensional
    self.assertEqual(prod.dim, 1)
    # check product computation
    result = prod([123], [456])
    self.assertEqual(result, [123 * 456])

  def test_complex(self):
    """ Complex algebra product test
    """
    # complex numbers may have different definitions
    definitions = [
        CliffordProduct((2, 0, 0), ["", "12"]),
        CliffordProduct((0, 1, 0), ["", "1"])
    ]
    for prod in definitions:
      # complex-valued algebra has two dimensions
      self.assertEqual(prod.dim, 2)
      # check product computation
      z1 = 12 + 45j
      z2 = -23 - 34j
      to_list = lambda z: [z.real, z.imag]
      result = prod(to_list(z1), to_list(z2))
      self.assertEqual(result, to_list(z1 * z2))

  def test_quaternion(self):
    """ Quaternion algebra product test
    """
    # complex numbers may have different definitions
    prod = CliffordProduct((3, 0, 0), ["", "12", "23", "13"])
    # quaternion algebra has four dimensions
    self.assertEqual(prod.dim, 4)
    # check product computation
    q1 = [1, 2, 3, 4]
    q2 = [-1, -2, -3, -4]
    ref = [28, -4, -6, -8]
    test = prod(q1, q2)
    self.assertEqual(ref, test)

  def test_convolution(self):
    """ Tests a convolution computation with CliffordProduct
    """
    # generate data
    from numpy.random import randint
    shape1=123,
    shape2=4
    z1 = randint(-5, 5, size=shape1) + randint(-5, 5, size=shape1) * 1j
    z2 = randint(-5, 5, size=shape2) + randint(-5, 5, size=shape2) * 1j
    # compute reference
    ref = numpy.convolve(z1, z2)
    # set up CliffordProduct for complex  numbers
    prod = CliffordProduct((2, 0, 0), ["", "12"])
    # compute the product
    z1 = [z1.real, z1.imag]
    z2 = [z2.real, z2.imag]
    test = prod.apply(lambda i, j : numpy.convolve(z1[i], z2[j]))
    # compare
    self.assertTrue((test[0] == ref.real).all())
    self.assertTrue((test[1] == ref.imag).all())


if __name__ == "__main__":
    unittest.main()