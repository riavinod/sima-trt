import os
import tensorflow as tf
import tensorflow.python.saved_model.signature_def_utils
from tensorflow.python.saved_model import signature_constants

trained_checkpoint_prefix = 'mobile/checkpoints/mobilenet_v1_1.0_224.ckpt'
export_dir = os.path.join('mobile_v1_model', '0')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta', clear_devices=True)
    loader.restore(sess, trained_checkpoint_prefix)
   
    
    #generate inputs
    inputs = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            inputs.append(op.name)
    print('Inputs:', inputs) 

    #signature map 
    #signature = signature_def_utils.build_signature_def(
    #        inputs = 
    
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()   

print('done')
