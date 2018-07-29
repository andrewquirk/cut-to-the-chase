from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from tldr import Encoder, TldrSystem, Decoder, Classifier
from os.path import join as pjoin

import logging

tf.app.flags.DEFINE_float("learning_rate", 10**-4, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 50.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropProb", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("numEpochs", 20, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("numClasses", 2, "The output size of your model.")
tf.app.flags.DEFINE_integer("glove_dim", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("train_directory", "data/dailymail_processed/train", "train directory")
tf.app.flags.DEFINE_string("dev_directory", "data/dailymail_processed/dev", "dev directory")
tf.app.flags.DEFINE_string("val_directory", "data/dailymail_processed/val", "val directory") # create a val directory from the first 100 samples in dev, use that to quickly test if things are working by changing train and dev directories to it as well
tf.app.flags.DEFINE_string("save_directory", "model", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_dir", "model", "Training directory to load model parameters from to resume training (default: {save_directory}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/vocab.dat", "Path to vocab file (default: ./data/vocab.dat)")
tf.app.flags.DEFINE_string("glove_path", "data/glove/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding (default: ./data/glove/glove.trimmed.{glove_dim}.npz)")
tf.app.flags.DEFINE_integer("clipGradients", 1, "1 = clip, 0 = don't clip")
tf.app.flags.DEFINE_integer("train_size", 176013, "train size") # I'm working on only one fourth of that
tf.app.flags.DEFINE_integer("dev_size", 21779, "dev size") # 21779
tf.app.flags.DEFINE_integer("maxSentenceLength", 100, "max num words in sentence")
FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def main(_):
    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)
    encoder = Encoder(FLAGS)
    decoder = Decoder(FLAGS)
    classifier = Classifier(FLAGS)
    tldr = TldrSystem(FLAGS, encoder, decoder, classifier)

    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    with tf.Session() as sess:
        initialize_model(sess, tldr, FLAGS.load_dir)
        tldr.train(sess, vocab, rev_vocab)
if __name__ == "__main__":
    tf.app.run()
    
    print(vars(FLAGS))
