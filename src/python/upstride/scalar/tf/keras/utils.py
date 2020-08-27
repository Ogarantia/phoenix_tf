# see code in https://math.stackexchange.com/questions/1103399/alternative-quaternion-multiplication-method
# FIXME: This file should not contain quaternion-related operations. They were copied from the folder for quaternions
# so that we would be able to run high-level tests.

def multiply_by_a(vector):
  """12 additions
  """
  A = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]

  output_vector = []
  for i in range(4):
    for j in range(4):
      if j == 0:
        if A[i][j] == 1:
          output_vector.append(vector[j])
        else:
          output_vector.append(-vector[j])
      else:
        if A[i][j] == 1:
          output_vector[i] = output_vector[i] + vector[j]
        else:
          output_vector[i] = output_vector[i] - vector[j]
  return output_vector


def quaternion_mult(tf_op, inputs, kernels, f=1):
  kernels = [k * f for k in kernels]
  if len(inputs) == 4:
    k1 = kernels[1] + kernels[2]
    k3 = kernels[0] + kernels[3]
    k4 = kernels[0] - kernels[3]
    k5 = kernels[1] - kernels[2]
    i1 = inputs[3] + inputs[1]
    i3 = inputs[0] - inputs[2]
    i4 = inputs[0] + inputs[2]
    i5 = inputs[3] - inputs[1]
    a1 = tf_op(i1, k1)
    a3 = tf_op(i3, k3)
    a4 = tf_op(i4, k4)
    a2 = a1 + a3 + a4
    a5 = 0.5*(a2 + tf_op(i5, k5))

    k1 = kernels[2] - kernels[3]
    k2 = kernels[1] + kernels[0]
    k3 = kernels[2] + kernels[3]
    k4 = kernels[0] - kernels[1]
    i1 = inputs[3] - inputs[2]
    i2 = inputs[1] + inputs[0]
    i3 = inputs[0] - inputs[1]
    i4 = inputs[3] + inputs[2]

    q1 = a5 - a1 + tf_op(i1, k1)
    q2 = a5 - a2 + tf_op(i2, k2)
    q3 = a5 - a3 + tf_op(i3, k3)
    q4 = a5 - a4 + tf_op(i4, k4)
    return [q1 * (1/f), q2 * (1/f), q3 * (1/f), q4 * (1/f)]
  else:
    outputs = [tf_op(inputs[0], kernels[i]) * (1/f) for i in range(4)]
  return outputs


def is_quaternion_init(init_type):
  """
  Determine whether it is a quaternion initialization or not
  Args:
      init_type: str or tf.keras.initializers.Initializer, initialization type for upstride quaternion, either
      'up2_init_he'  or 'up2_init_glorot' for real valued initialization should be tensorflow
  """

  if isinstance(init_type, str) and 'up2_init' in init_type:
    return True

  return False
