import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def covert_pb_to_server_model(pb_model_path, export_dir, input_name='input', output_name='output'):
    graph_def = read_pb_model(pb_model_path)
    covert_pb_saved_model(graph_def, export_dir, input_name, output_name)
def read_pb_model(pb_model_path):
    #pb_model_path.decode("utf-8")
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def
def covert_pb_saved_model(graph_def, export_dir, input_name='input:0', output_name='output'):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_name = 'input:0'
        #output_name = 'MobilenetV1/Predictions/Softmax:0'
        out_nodes = [n.name  for n in graph_def.node if n.op in ('Softmax') or n.op in ('ArgMax')]
        for n in graph_def.node:
            print(n.name)
        print(out_nodes)
        output_name = out_nodes[0] + ':0'
        #output_name = 'squeezed' + ':0'
        input_nodes = [n.name for n in graph_def.node if n.op in ('Placeholder')]
        print('input name:', input_nodes)
        print('output name:', output_name)
        
        input_name = input_nodes[0] + ':0'
        
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)
        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})
        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()
covert_pb_to_server_model('inc/saved_model.pb', 'inc_serve_pb')
#covert_pb_to_server_model('pb_files/inception_v2_converted.pb', 'inception')
#covert_pb_to_server_model('mobilev2_pb/saved_model.pb', 'mobilev2_serve_pb')
