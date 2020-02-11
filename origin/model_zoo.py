import tensorflow as tf
import i3d
import re

class I3DNet:
    def __init__(self, inps, pretrained_model_path, final_end_point, trainable=False, scope='v/SenseTime_I3D'):

        self.final_end_point = final_end_point
        self.trainable = trainable
        self.scope = scope

        # build entire pretrained networks (dummy operation!)
        i3d.I3D(inps, scope=scope, is_training=trainable)

        var_dict = { re.sub(r':\d*','',v.name):v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D') }
        self.assign_ops = []
        for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
            # load variable
            var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
            assign_op = var_dict[var_name].assign(var)
            self.assign_ops.append(assign_op)

    def __call__(self, inputs):
        out, _ = i3d.I3D(inputs, final_endpoint=self.final_end_point, scope=self.scope, is_training=self.trainable, reuse=True)
        return out