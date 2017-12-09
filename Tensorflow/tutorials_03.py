# Scopes and when to use them
"""
Variables and tensors in TensorFlow have a name
attribute that is used to identify them in the
symbolic graph.
If you don't specify a name when creating a variable
or a tensor, TensorFlow automatically assigns a name for you:
"""

import tensorflow as tf

a = tf.constant(1)
print(a.name)  # Const:0

b = tf.Variable(1)
print(b.name)  # Variable:0

# You can overwrite the default name by explicitly specifying it:

a = tf.constant(1, name="c_a")
print(a.name)  # c_a:0

b = tf.Variable(1, name="V_b")
print(b.name)  # V_b:0

# TensorFlow introduces two different context managers to alter
# the name of tensors and variables. The first is tf.name_scope:
with tf.name_scope("name_scope"):
    a = tf.constant(1, name="a")
    print(a.name)  # name_scope/a:0

    b = tf.Variable(1, name="b")
    print(b.name)  # name_scope/b:0

    c = tf.get_variable(name="c", shape=[])
    print(c.name)  # c:0

# Note that there are two ways to define new variables in TensorFlow,
# by creating a tf.Variable object or by calling tf.get_variable.
# Calling tf.get_variable with a new name results in creating a new variable,
# but if a variable with the same name exists it will raise a ValueError exception,
# telling us that re-declaring a variable is not allowed.
# tf.name_scope affects the name of tensors and variables created with tf.Variable,
# but doesn't impact the variables created with tf.get_variable.

# Unlike tf.name_scope, tf.variable_scope modifies the name of variables created with tf.get_variable as well:
with tf.variable_scope("variable_scope"):
    a = tf.constant(1, name="a")
    print(a.name)  # variable_scope/a:0

    b = tf.Variable(1, name="b")
    print(b.name)  # variable_scope/b:0

    c = tf.get_variable(name="c", shape=[])
    print(c.name)  # variable_scope/c:0

with tf.variable_scope("variable_scope_02"):
    a1 = tf.get_variable(name="a", shape=[])
    # a2 = tf.get_variable(name="a", shape=[])  # Disallow

# But what if we actually want to reuse a previously declared variable?
# Variable scopes also provide the functionality to do that:
with tf.variable_scope("scope"):
    a1 = tf.get_variable(name="a", shape=[])
with tf.variable_scope("scope", reuse=True):
    a2 = tf.get_variable(name="a", shape=[])
    print(a2.name)

# This becomes handy for example when using built-in neural network layers:
features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
# Use the same convolution weights to process the second image:
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)

# TensorFlow templates are designed to handle this automatically:
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.

# You can turn any function to a TensorFlow template. Upon the first call to a template,
# the variables defined inside the function would be declared and in the consecutive invocations
# they would automatically get reused.