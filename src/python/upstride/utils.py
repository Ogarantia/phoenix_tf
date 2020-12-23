import functools

@functools.lru_cache(maxsize=32) # Caching results to avoid recomputing when not needed
def permutation(source : str, destination : str):
  """ Returns a list containing the permutation needed to go from the 'source' to the 'destination'.
  """
  assert len(source) == len(destination)
  for i in range(len(source) - 1):
    if source[i] in source[i + 1:] or destination[i] in destination[i + 1:]:
      raise ValueError("Both 'source' and 'destination' are expected to contain no more than 1 occurance of each character.")
  permutation = [0]*len(source)
  try:
    for i in range(len(source)):
      permutation[i] = source.index(destination[i])
  except:
    raise ValueError("One input is expected to be a permutation of the other.")
  return permutation


def listify(x):
  """ If x is a list, returns it as is. Otherwise returns a list containing x itself. """
  return x if isinstance(x, list) else [x]