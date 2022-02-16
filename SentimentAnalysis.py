# The Thankful AI Co. by Thomas Chia
# Sentiment Analysis Program

import data_manager
import argparse
import tensorflow as tf
import os
from gooey import Gooey, GooeyParser

# tf.compat.v1.disable_v2_behavior()

# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

working_dir = os.getcwd()
checkpoint_path = "model.ckpt"
meta_graph_path = "model.ckpt.meta"
config_path = "config.pkl"
stopwords_path = "stopwords.txt"

def read_files(file_name):
    """Reads each file line by line."""
    with open(file_name, mode='r') as f:
        file_contents = f.read().splitlines()
    return file_contents

def run_SA(data):
    """Tests the sentiment of the data."""
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        # Import graph and restore its weights
        saver = tf.compat.v1.train.import_meta_graph(meta_graph_path)
        saver.restore(sess, checkpoint_path)
        # Recover input/output tensors
        input = graph.get_operation_by_name('input').outputs[0]
        seq_len = graph.get_operation_by_name('lengths').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            'dropout_keep_prob').outputs[0]
        predict = graph.get_operation_by_name(
            'final_layer/softmax/predictions').outputs[0]
        # Perform prediction
        pred = sess.run(
            [predict],
            feed_dict={
                input: data[0],
                seq_len: data[1],
                dropout_keep_prob: 1})
        return pred

def process_data(path_to_data):
    """Processes data from a text file."""
    data_arr = read_files(path_to_data)
    ds = data_manager.DataManager(
        data_arr = data_arr,
        stopwords_file = stopwords_path)
    original_text, processed_text, seq_length = ds.data(original_text=True)
    print(original_text)
    return [processed_text, seq_length, original_text]

def process_predictions(predictions, original_text):
    output_file_data = []
    for sample in range(len(original_text)):
        sentiment = float(predictions[0][sample, 1])
        pos_or_neg = False
        if sentiment > float(.60):
            pos_or_neg = True
        if pos_or_neg:
            feeling = "Positive"
        else:
            feeling = "Negative"
        output_file_data.append("Input sample: " + str(original_text[sample]))
        output_file_data.append(
            "Predicted sentiment: " + feeling)
        output_file_data.append("\n")
    return output_file_data

def write_to_file(outputs, output_dir):
    output_file = os.path.join(output_dir, "sentiment_predictions.txt")
    print("File saved to: ", output_file)
    with open(output_file, 'w') as f:
        for item in outputs:
            f.write("%s\n" % item)

@Gooey(    
    dump_build_config = False,
    program_name = "Sentiment Analysis Suite")
def arguments():
    parser = GooeyParser(
        description = "The Thankful AI Co.: Sentiment Analysis Suite")
    parser_group = parser.add_argument_group(
        "Select files")
    parser_group.add_argument(
        "--Inputfile",
         "-i",
         help = "Path to the input file with sentences.", 
         widget = "FileChooser")
    parser_group.add_argument(
        "--Outputdir",
        "-o",
        help = "Path to the save folder.",
        widget = "DirChooser")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguments()
    data = process_data(path_to_data=args.Inputfile)
    predictions = run_SA(data = data)
    processed_predictions = process_predictions(
        predictions = predictions, original_text = data[2])
    write_to_file(
        outputs = processed_predictions, 
        output_dir = args.Outputdir)

    