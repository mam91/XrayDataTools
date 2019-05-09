import tensorflow as tf
import dataset_util
import xray_data
import io

flags = tf.app.flags
flags.DEFINE_string('output_path', 'C:/Users/mmill/Documents/PythonProjects/XrayDataTools/output/xray_train.record', 'Path to output TFRecord')
flags.DEFINE_string('input_path', 'C:/Users/mmill/Documents/PythonProjects/XrayDataTools/input', 'Path to input directory')
FLAGS = flags.FLAGS

def create_tf_example(image_data):
    height = image_data.height # Image height
    width = image_data.width # Image width

    with tf.gfile.GFile(image_data.path, 'rb') as fid:
        encoded_png = fid.read()

    filename = image_data.id.encode('utf8')

    image_format = b'png' # b'jpeg' or b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for bbox in image_data.bboxes:
        xmins.append(bbox.min_x / width)
        xmaxs.append(bbox.max_x / width)
        ymins.append(bbox.min_y / height)
        ymaxs.append(bbox.max_y / height)
        classes_text.append(bbox.class_label.encode('utf8'))
        classes.append(bbox.class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
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
    dset.load_data(FLAGS.input_path)

    for i in range(len(dset.images)):
        tf_example = create_tf_example(dset.images[i])
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Successfully converted dataset to tf record")

if __name__ == '__main__':
    tf.app.run()