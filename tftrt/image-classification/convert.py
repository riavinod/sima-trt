import tensorflow as tf

#with tf.Session() as sess:

        # Restore the graph from (.meta .data .index)
        #saver = tf.train.import_meta_graph(f"{checkpoint_path}/{meta_file_string}")
        #saver.restore(sess, tf.train.latest_checkpoint(str(checkpoint_path)))


        # Convert into ".pb" using SavedModel API.
model_path = f'resnet101'
builder = tf.saved_model.builder.SavedModelBuilder(model_path)

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.SERVING],
    main_op=tf.tables_initializer(),
    strip_default_attrs=True)

builder.save()
print("Saved")
