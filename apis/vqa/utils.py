import numpy as np
from scipy.misc import imresize, imread
import tensorflow as tf


def extract_fc7_features(img, model_path):
    vgg_file = open(model_path, "rb")
    vgg16raw = vgg_file.read()
    vgg_file.close()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(vgg16raw)
    images = tf.placeholder("float32", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images": images})
    graph = tf.get_default_graph()
    sess = tf.Session()

    img = imread(img)
    img_resized = imresize(img, (224, 224))
    image_array = (img_resized / 255.0).astype("float32")
    image_feed = np.ndarray((1, 224, 224, 3))
    image_feed[0:, :, :] = image_array
    feed_dict = {images: image_feed}
    fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
    fc7_features = sess.run(fc7_tensor, feed_dict=feed_dict)
    sess.close()
    return fc7_features
