# understanding static and dynamic shapes
import tensorflow as tf

a = tf.placeholder(tf.float32, [None, 128])

# This means that the first dimension can be of any size and
# will be determined dynamically during Session.run().
# You can query the static shape of a Tensor as follows:
static_shape = a.shape.as_list()
print("static_shape:", static_shape)

# To get the dynamic shape of the tensor you can call tf.shape op,
# which returns a tensor representing the shape of the given tensor:
dynamic_shape = tf.shape(a)
print("dynamic_shape:", dynamic_shape)

# The static shape of a tensor can be set with Tensor.set_shape() method:
a.set_shape([32, 128])
a.set_shape([None, 128])

# You can reshape a given tensor dynamically using tf.reshape function:
a = tf.reshape(a, [32, 128])

# It can be convenient to have a function that returns the static shape
# when available and dynamic shape when it's not.
# The following utility function does just that:
def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
    return dims


# Now imagine we want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing
# the second and third dimensions into one.
# We can use our get_shape() function to do that:
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
b = tf.reshape(b, [shape[0], shape[1]*shape[2]])




