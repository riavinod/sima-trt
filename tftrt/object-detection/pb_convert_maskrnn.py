import os
import shutil
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
def build_saved_model(build_sess, output_dir):
    # Print all nodes found
    print('Graph nodes:\n')
#    for op in build_sess.graph.get_operations():
#        print(op.name)
    # Generate the servable builder
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_dir)
    input_op_ph1 = build_sess.graph.get_operation_by_name('input_image')#'Placeholder_1')
    input_op_ph2 = build_sess.graph.get_operation_by_name('input_image_meta')
    input_op_ph3 = build_sess.graph.get_operation_by_name('input_anchors')
    output_op = build_sess.graph.get_operation_by_name('mrcnn_class/Softmax')
    input_tensor_ph1 = input_op_ph1.outputs[0]
    input_tensor_ph2 = input_op_ph2.outputs[0]
    input_tensor_ph3 = input_op_ph2.outputs[0]
    output_tensor = output_op.outputs[0]
    info_ph1 = tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor_ph1)
    info_ph2 = tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor_ph2)
    info_ph3 = tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor_ph2)
    info_sm  = tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)
    signature = tf.compat.v1.saved_model.build_signature_def(
        inputs={'imput_images':info_ph1,
                 'input_image_meta':info_ph2,
                 'input_anchors':info_ph3
               }, outputs={'softmax':info_sm},
        method_name=tf.compat.v1.saved_model.PREDICT_OUTPUTS)
    # Add one model and its variables preserving inputs and outputs
    builder.add_meta_graph_and_variables(build_sess,
                                        tags=[tf.saved_model.SERVING],
                                        signature_def_map={'serving_default':signature},
                                        strip_default_attrs=True)
    builder.save()
def load_frozen_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print("placeholder: ", [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')])
        print("output: ", [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Softmax')])
        print("output: ", [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Argmax')])

    # Then, we import the graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph

#%%
# We use our "load_frozen_graph" function
graph = load_frozen_graph('maskrcnn.pb')

#%%
# We launch a Session - /0 in output path is important
with tf.Session(graph=graph) as sess:
    build_saved_model(sess, 'savedMovel/0')
