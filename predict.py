from __future__ import print_function
from data_load import basic_tokenizer
import codecs
import os
from data_load import load_test_data2, load_source_vocab, load_target_vocab
from models import *
from tqdm import tqdm
import math

def predict(sentences):
    # Load graph
    g = TransformerDecoder(is_training=False)
    print("Graph loaded")

    # Load data
    X, sources, actual_lengths = load_test_data2(sentences)

    sorted_lengths = np.argsort(actual_lengths)
    X = X[sorted_lengths]
    print(X.shape)

    src2idx, idx2src = load_source_vocab()
    tgt2idx, idx2tgt = load_target_vocab()

    # Start session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(graph=g.graph, config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        print("Restored!")
 
        batch_size = hp.batch_size
        num_batches = math.ceil(len(X)/batch_size)
        print("Num batch size", num_batches)
        Y_preds = np.zeros_like(X) + 2

        for i in tqdm(range(num_batches), desc="Inference: "):
            indices = np.arange(i*batch_size, min((i+1)*batch_size, len(X)))
            print("indices", indices)
            step = 0
            max_steps = math.ceil((np.max(actual_lengths[indices]) - hp.offset)/(hp.maxlen - hp.offset))
            max_steps = max(1, max_steps)
            print("Max step", max_steps)
            for step in range(max_steps):
                end = min(step*(hp.maxlen - hp.offset) + hp.maxlen, max(X.shape[1], hp.maxlen))
                start = end - hp.maxlen

                print("Shape X[1]", end)

                x = X[indices, start: end]
                print("XXXXXXXXXXX")
                print(X)
                print("xxxxxxxxxx")
                print(x.shape)
                _preds = sess.run(g.preds, {g.x: x, g.dropout: False})
                if step > 0:
                    Y_preds[indices, start+hp.offset//2:end] = _preds[:, hp.offset//2:]
                else:
                    Y_preds[indices, start:end] = _preds

        Y_preds = Y_preds[np.argsort(sorted_lengths)]
        result = ""
        for source, preds, actual_length in zip(sources, Y_preds, actual_lengths):
            formatted_pred = [idx2tgt[idx] if src2idx.get(source[id],1) > 8 else source[id] for id, idx in enumerate(preds[:actual_length])] 
            print(formatted_pred)  
            sentence = " ".join(formatted_pred)
            result = result + " " + sentence
        return result