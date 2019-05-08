import tensorflow as tf
import dataset_util
import xray_data
import cv2

flags = tf.app.flags
flags.DEFINE_string('output_path', 'C:/Users/mmill/Documents/PythonProjects/XrayDataTools/output/xray_tf_format', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(image_data):
    # TODO(user): Populate the following variables from your example.
    height = image_data.height # Image height
    width = image_data.width # Image width
    filename = str.encode(image_data.id) # Filename of the image. Empty if image is not from file

    image_bytes = cv2.imread(image_data.path, 0)
    encoded_image_data = image_bytes.tobytes() # Encoded image bytes

    image_format = b'png' # b'jpeg' or b'png'

    xmins = [image_data.bboxes[0].min_x] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [image_data.bboxes[0].max_x] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [image_data.bboxes[0].min_y] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [image_data.bboxes[0].max_y] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [str.encode(image_data.label)] # List of string class name of bounding box (1 per box)
    
    classes = [image_data.bboxes[0].class_id] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable
    dset = xray_data.XrayDataset()
    dset.load_data('./input')

    for i in range(len(dset.images)):
        tf_example = create_tf_example(dset.images[i])
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()