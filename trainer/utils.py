from tensorflow.python.lib.io import file_io
import os
import sys
import urllib
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  print("Copying %s to %s" % (job_dir, file_path))
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
      output_f.write(input_f.read())


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(
      inputs={'input': model.inputs[0]}, outputs={'output': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        })
    builder.save()


# Ripped off from https://github.com/chen0040/keras-chatbot-web-api/blob/master/chatbot_train/cornell_word_seq2seq_glove_train.py
def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


# Ripped off from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
  
  if not tf.gfile.Exists(work_directory):
    tf.gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  print("Downloading %s to %s" % (source_url, filepath))
  if not tf.gfile.Exists(filepath):
    temp_file_name, _ = urllib.request.urlretrieve(url=source_url, reporthook=reporthook)
    print("tmpfile", temp_file_name)
    tf.gfile.Copy(temp_file_name, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath