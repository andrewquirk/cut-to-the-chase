from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import datetime
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
# I: Q: I forgot to tell you guys, I split the first 100 samples in dev set as a validation set that we can use for sanity checking
# taking the first 100 is not the best thing to do, but it was the fastest way for me at that time..
# also just a reminder, we can decrease our dataset sizes and work on a subset of the training set 
# if things don't look well as the deadline approaches

logging.basicConfig(level=logging.INFO)

class Encoder(object):
    # COMMENTS ARE IRRELEVANT
    # initializer=tf.uniform_unit_scaling_initializer(1.0)
    def __init__(self,FLAGS):
        self.state_size = FLAGS.state_size
        self.maxSentenceLength = FLAGS.maxSentenceLength
        # Q: these will be BasicLSTMCells
        # self.cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        # self.cell2 = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        with vs.variable_scope("encoder", initializer = tf.contrib.layers.xavier_initializer()):
            self.W = tf.get_variable("W", dtype = tf.float64, shape = (self.state_size, self.state_size))
            self.b = tf.get_variable("b", dtype = tf.float64, shape = (self.state_size,), initializer = tf.zeros_initializer())
        # w is state_size by state_size
    def encode_sentences(self, article, seqlen):
        # article is numsentences by maxSentenceLength by glove_dim
        # Q: if we want a BiLSTM, this is going to be bidirectional_dynamic_rnn, and similarly for the others as well
        article_reshaped = tf.reshape(article, [-1, self.state_size])
        h = tf.nn.relu(tf.matmul(article_reshaped, self.W) + self.b)
        h = tf.reshape(h, [-1, self.maxSentenceLength, self.state_size])
        # o is numsentences by maxSentenceLength by stateSize
        # h is numsentences by stateSize
        return h
    # def encode_article(self, article, h_sentences):
        # maybe add a relu before
        # h_expanded = tf.expand_dims(h_sentences, axis = 0)
        # print(self.cell)
        # with vs.variable_scope("article"):
            # _, h_article = tf.nn.dynamic_rnn(self.cell2, h_expanded, dtype = tf.float64)
            # h_article = tf.reduce_sum(h_article, axis = 0)
        # return h_article
    # def attention_encode(self, o, h_article):
        # h_article is 1 by 1 by stateSize
        # o_reshaped = tf.reshape(o, [-1, self.state_size])
        # z = tf.matmul(o_reshaped, self.W) 
        # z is numSentences by maxSentenceLength by stateSize
        # scores = tf.matmul(z, h_article, transpose_b = True)
        # print(scores)
        # scores = tf.reshape(scores, [-1, self.maxSentenceLength, 1])
        # scores is numSentences by maxSentenceLength by 1
        # o_final = tf.multiply(o, scores)
        # o_final has same size as o
        # return o_final

class Decoder(object):
    def __init__(self,FLAGS):
        # Q: we can use an LSTM in the decoder too, but it may be a better idea not to increase the number of parameters too much
        self.state_size = FLAGS.state_size
        self.maxSentenceLength = FLAGS.maxSentenceLength
        with vs.variable_scope("decoder", initializer = tf.contrib.layers.xavier_initializer()):
            self.W = tf.get_variable("W", dtype = tf.float64, shape = (self.state_size,1))
            self.b = tf.get_variable("b", dtype = tf.float64, shape = (1,),
            initializer = tf.zeros_initializer())
    def decode(self,h):
        # print(o)
        # o_reshaped = tf.reshape(o, [-1, self.state_size])
        # z = tf.matmul(o_reshaped,self.W) + self.b
        # z = tf.reshape(z, [-1, self.maxSentenceLength, 1])
        # z = tf.reduce_sum(z, axis = 2)
        print('h')
        print(h)
        print('h')
        h_reshaped = tf.reshape(h, [-1, self.state_size])
        z = tf.matmul(h_reshaped, self.W) + self.b
        z = tf.reshape(z, [-1, self.maxSentenceLength, 1])
        z = tf.reduce_sum(z, axis = 2)
        return tf.nn.relu(z)

class Classifier(object):
    def __init__(self, FLAGS):
        self.numClasses = FLAGS.numClasses
        self.maxSentenceLength = FLAGS.maxSentenceLength   
        with vs.variable_scope("classifier", initializer = tf.contrib.layers.xavier_initializer()):
            # self.U = tf.get_variable("U", dtype = tf.float64, 
            # shape = (self.maxSentenceLength,self.numClasses))
            self.U = tf.get_variable("U", dtype = tf.float64, 
            shape = (self.maxSentenceLength,1))
            # self.b = tf.get_variable("b", dtype = tf.float64, shape = (self.numClasses,), 
            # initializer=tf.uniform_unit_scaling_initializer(1.0))
            self.b = tf.get_variable("b", dtype = tf.float64, shape = (1,), 
            initializer = tf.zeros_initializer())

    def classify(self, h):
        # print(h)
        return tf.matmul(h, self.U) + self.b

# class Classifier(object):
    # def __init__(self, FLAGS):
        # self.numClasses = FLAGS.numClasses
        # self.maxSentenceLength = FLAGS.maxSentenceLength   
        # with vs.variable_scope("classifier", initializer = tf.contrib.layers.xavier_initializer()):
            # self.U = tf.get_variable("U", dtype = tf.float64, 
            # shape = (self.maxSentenceLength,self.numClasses))
            # self.b = tf.get_variable("b", dtype = tf.float64, shape = (self.numClasses,), 
            # initializer=tf.uniform_unit_scaling_initializer(1.0))

    # def classify(self, h):
        # print(h)
        # return tf.matmul(h, self.U) + self.b

class TldrSystem(object):

    def __init__(self, FLAGS, encoder, decoder, classifier):
        
        self.FLAGS = FLAGS
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.xplaceholder = tf.placeholder(tf.int32, 
                            shape = (None, self.FLAGS.maxSentenceLength))
        self.yplaceholder = tf.placeholder(tf.float64, shape = (None,)) 
        # self.maskplaceholder = tf.placeholder(tf.int32, shape = (None,self.FLAGS.maxSentenceLength))
        self.maskplaceholder = tf.placeholder(tf.int32, shape = (None,))
        self.drop_placeholder = tf.placeholder(tf.float64, shape = ())
        self.lr_placeholder = tf.placeholder(tf.float64, shape = ())
        self.opplaceholder = tf.placeholder(tf.float64)
        with tf.variable_scope("tldr", initializer = tf.contrib.layers.xavier_initializer()):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
        
        params = tf.trainable_variables()
        self.globalnorm = 0
        self.paramnorm = 0
        for param in params:
            shp = param.get_shape()
            if len(shp) >= 2:
                self.paramnorm += tf.nn.l2_loss(param)
        opt = tf.train.AdamOptimizer(self.lr_placeholder)
        if self.FLAGS.clipGradients == 1:	
                try:
                    grads, _ = zip(*opt.compute_gradients(self.loss))
                    grads, _ =  tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)
                    self.globalnorm = tf.global_norm(grads)
                    grads_vars = zip(grads, params)
                    self.updates = opt.apply_gradients(grads_vars)
                except AttributeError:
                    self.updates = None
        else:
            grads = tf.gradients(self.loss, params)
            self.globalnorm = tf.global_norm(grads)
            try:
                self.updates = opt.minimize(self.loss)
            except AttributeError:
                self.updates = None
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 2, max_to_keep = 0)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            glove_matrix = tf.constant(np.load(self.FLAGS.glove_path)['glove'])
            self.x = tf.nn.embedding_lookup(glove_matrix, self.xplaceholder)


    def setup_system(self):
        # full version, can revert to simpler version and see how they compare
        # o, h_sentences = self.encoder.encode_sentences(self.x, self.maskplaceholder)
        # h_article = self.encoder.encode_article(self.x, h_sentences)
        # o_drop = tf.nn.dropout(o, self.drop_placeholder)
        # h_article_drop = tf.nn.dropout(h_article, self.drop_placeholder)
        # o_final = self.encoder.attention_encode(o, h_article)
        # o_final_drop = tf.nn.dropout(o_final, self.drop_placeholder)
        # h = self.decoder.decode(o_drop)
        # h = self.decoder.decode(o)
        # h = self.decoder.decode(o_final_drop)
        # h = self.decoder.decode(o_final)
        # h_drop = tf.nn.dropout(h, self.drop_placeholder)
        h1 = self.encoder.encode_sentences(self.x, self.maskplaceholder)
        h1_drop = tf.nn.dropout(h1, self.drop_placeholder)
        h2 = self.decoder.decode(h1_drop)
        h2_drop = tf.nn.dropout(h2, self.drop_placeholder)
        self.pred = self.classifier.classify(h2_drop)
        # self.pred = self.classifier.classify(h)
        print(self.pred)

    def setup_loss(self):
        with vs.variable_scope("loss"):
            # l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.pred, 
            # labels = self.yplaceholder)
            # self.loss = tf.reduce_sum(l)
            # if self.lr_placeholder is not None:
            l = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.pred,
            labels = tf.expand_dims(self.yplaceholder, 1))
            # l = [tf.float64(0.0)]
            self.loss = tf.matmul(l, tf.expand_dims(self.opplaceholder, 1), transpose_a = True)
                # one_inds = []
                # num_sents = tf.shape(self.yplaceholder)[0]
                # for i in range(self.opplaceholder):
                    # if self.yplaceholder[i] == 1:
                        # one_inds.append(i)
                # num_ones = len(one_inds)
                # prob_random =  2.0 * len(one_inds) / len(self.yplaceholder)
                # one_labels = self.yplaceholder[one_inds]
                # one_logits = self.pred[one_inds][:]
                # random_inds = tf.random_shuffle([i for i in range(tf.opplaceholder)])
                # random_labels = self.yplaceholder[random_inds[0:2*num_ones]]
                # random_logits = self.pred[random_inds[0:2*num_ones]][:]
                # l = tf.nn.sigmoid.cross_entropy_with_logits(logits = tf.concat([one_logits, random_logits], 0),
                # labels = tf.expand_dims(tf.concat([one_labels, random_labels], 0), 1))
            # else:
            # l = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.pred, 
            # labels = tf.expand_dims(self.yplaceholder, 1))
            # self.loss = tf.reduce_sum(l)
            # maybe switch back to sum later

    def optimize(self, sess, train_x, train_y, train_mask, drop, lr, chosen_inds):
        input_feed = {}
        input_feed[self.xplaceholder] = train_x
        input_feed[self.yplaceholder] = train_y
        input_feed[self.maskplaceholder] = train_mask
        input_feed[self.drop_placeholder] = drop
        input_feed[self.lr_placeholder] = lr
        input_feed[self.opplaceholder] = chosen_inds
        output_feed = [self.updates, self.globalnorm, self.paramnorm, self.loss]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def test(self, sess, dev_x, dev_y, dev_mask, chosen_inds, drop=1.0, lr = None):
        input_feed = {}
        input_feed[self.xplaceholder] = dev_x
        input_feed[self.yplaceholder] = dev_y
        input_feed[self.maskplaceholder] = dev_mask
        input_feed[self.drop_placeholder] = drop
        # input_feed[self.lr_placeholder] = None
        input_feed[self.opplaceholder] = chosen_inds
        output_feed = [self.loss]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def answer(self, sess, dev_x, dev_y, dev_mask, drop=1.0, lr = None):
        input_feed = {}
        input_feed[self.xplaceholder] = dev_x
        input_feed[self.yplaceholder] = dev_y
        input_feed[self.maskplaceholder] = dev_mask
        input_feed[self.drop_placeholder] = drop
        # input_feed[self.lr_placeholder] = None
        # input_feed[self.opplaceholder] = chosen_inds
        output_feed = [self.pred]
        outputs = sess.run(output_feed,input_feed)
        return outputs
    # def generateSummary(self, sess, sentences, dev_x, dev_y, dev_mask):
        # outputs = self.answer(sess, dev_x, dev_y, dev_mask)
        # preds = outputs[0]
        # print(len(preds))
        # print(len(sentences))
        # summary = []
        # error = 0 
        # print(dev_y)
        # for i in range(len(preds)):
            # pred = preds[i]
            # print(pred)
            # if pred[1] > pred[0]:
                # print(sentences[i])
                # summary.append(sentences[i])
                # if dev_y[i] == 0:
                    # error += 1
            # elif dev_y[i] == 1:
                # error += 1
        # print(error)
        # accuracy = 1.0 - (error * 1.0) / len(preds)
        # print(accuracy)
        # print(summary)
        # return summary, accuracy
        # raise NotImplementedError

    def generateSummary(self, sess, sentences, dev_x, dev_y, dev_mask):
        outputs = self.answer(sess, dev_x, dev_y, dev_mask)
        preds = outputs[0]
        # print(len(preds))
        # print(len(sentences))
        summary = []
        error = 0 
        # print(dev_y)
        for i in range(len(preds)):
            pred = preds[i]
            # print(pred)
            # if pred[1] > pred[0]:
            if pred > 0:
                # print(sentences[i])
                summary.append(sentences[i])
                if dev_y[i] == 0:
                    error += 1
            elif dev_y[i] == 1:
                error += 1
        # print(error)
        accuracy = 1.0 - (error * 1.0) / len(preds)
        # print(accuracy)
        # print(summary)
        return summary, accuracy

    def rouge_2(self, sess, sentences, dev_x, dev_y, dev_mask, summary):
        # take the prediction from self.answer and compute rouge_2 score on one article from dev set
        # using Ian's script for that
        generatedSummary, accuracy = self.generateSummary(sess, sentences, dev_x, dev_y, dev_mask)
        # print(summary)
        # print(summary[1:])
        if len(generatedSummary) == 0: return 0, accuracy
        strSummary = []
        for sent in generatedSummary:
            for word in sent:
                strSummary.append(word)
        # print(summary)
        summary = summary.split(' ')
        # print(summary)
        # summary = summary.split
        # outputs = ' '.join(self.answer(dev_x, dev_y, dev_mask))
        # outputs =  ' '.join(generatedSummary)
        refBigrams = set(zip(summary, summary[1:]))
        # print(refBigrams)
        # outputs = ''
        summaryBigrams = set(zip(strSummary, strSummary[1:]))

        if len(summaryBigrams) == 0 or len(refBigrams) == 0:
            return 0, accuracy

        count = 0
        for bigram in refBigrams:
            if bigram in summaryBigrams:
                count += 1
        if count == 0: return 0, accuracy
        count1 = (count * 1.0) / len(refBigrams)
        count2 = (count * 1.0) / len(summaryBigrams)
        f1 = 2 * (count1 * count2) / (count1 + count2)


        return f1, accuracy    
        
    # def rouge_2(self, sess, sentences, dev_x, dev_y, dev_mask):
        # take the prediction from self.answer and compute rouge_2 score on one article from dev set
        # using Ian's script for that
        # outputs = self.answer(dev_x, dev_y, dev_mask)
        # generatedSummary = self.generateSummary(self, sess, sentences, dev_x, dev_y, dev_mask)
        # I: this summary is a list of sentences, so you would need to turn it into a string if necessary
        # also for some articles the summary can be empty (especially in the beginning I guess) so you may want
        # to keep that in mind to avoid bugs
        # raise NotImplementedError
        # TO BE IMPLEMENTED

    def readPreprocessedData(self, dirName, batchSize=1):

        data = []

        for fileName in os.listdir(dirName):
            if fileName == ".DS_Store": continue
            with open(os.path.join(dirName, fileName), 'r') as f:

                sentences = []
                summaryFlags = []

                line = f.readline().strip()
                while '@summary' not in line:

                    if len(line) == 0:
                        continue

                    parts = line.split()

                    sentences.append(parts[:-1])
                    summaryFlags.append(int(parts[-1]))

                    line = f.readline().strip()

            #Third part of the tuple is the summary as given in the dataset
                data.append((sentences, summaryFlags, f.readline().strip()))

            if len(data) == batchSize:
                yield data
                data = []

        yield data    


    # def readPreprocessedData(self, dirName, batchSize=1):

      


      #   data = []
        # for fileName in os.listdir(dirName):
          #   if fileName == ".DS_Store": continue
            # with open(os.path.join(dirName, fileName), 'r') as f:

              #   sentences = []
              #   summaryFlags = []
                # for line in f:

                  #   if len(line) == 0:
                    #     continue

                    # parts = line.split()
                    # if len(parts) < 2:
                    #     continue
                    # sentences.append(parts[:len(parts)-1])
                    # summaryFlags.append(parts[-1])

            # data.append((sentences, summaryFlags))

            # if len(data) == batchSize:
              #   yield data
               #  data = []
        # yield data

    def pad_n_mask(self, sentences):
        numSentences = len(sentences)
        maskedSentences = np.zeros(shape = (numSentences, self.FLAGS.maxSentenceLength), dtype = 
        np.int32)
        # mask = np.zeros(shape = (numSentences, self.FLAGS.maxSentenceLength), dtype=np.int32)
        mask = []
        for i in range(numSentences):
            senLength = len(sentences[i])
            if senLength < self.FLAGS.maxSentenceLength:
                # mask[i] = ([1] * senLength + [0] * (self.FLAGS.maxSentenceLength - senLength))
                mask.append(senLength)
                maskedSentences[i] = (sentences[i] + [0] * (self.FLAGS.maxSentenceLength - senLength))
            else:
                # mask[i] = [1] * self.FLAGS.maxSentenceLength
                mask.append(self.FLAGS.maxSentenceLength)
                maskedSentences[i] = (sentences[i][:self.FLAGS.maxSentenceLength])
        return maskedSentences, mask

    def basic_tokenizer(self, sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(re.split(" ", space_separated_fragment))
        return [w for w in words if w]    

    def sentence_to_token_ids(self, sentence, vocabulary, tokenizer=None):
        # if tokenizer:
            # words = tokenizer(sentence)
        # else:
            # words = self.basic_tokenizer(sentence)
        # UNK_ID = 2
        return [vocabulary.get(w, 2) for w in sentence]

    def words_to_inds(self, sentences, vocab):
        sentInds = []
        for sentence in sentences:
            sentInds.append(self.sentence_to_token_ids(sentence, vocab))
        return sentInds

    def train(self, sess, vocab, rev_vocab):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))
        path = "./model/checkpoint"
        num = -1
        if os.path.exists(path):
            f = open(path,"r") 
            line = file.readline(f)
            end = line[-4:]
            for i in range(len(end)):
                if end[i] == '-': break
            num = end[i+1:-2]
        # print(end[-1]) 
        num = int(num)
        # print(num)
        for e in range(num+1, self.FLAGS.numEpochs, 1):

            
            # print(e, num)
            # if e <= num: continue
            current_lr = self.FLAGS.learning_rate / (2**e)
            i = 0
            average_time = 0
            for data in self.readPreprocessedData(self.FLAGS.train_directory):
                i += 1
                if i > self.FLAGS.train_size/4: break
                if len(data) == 0:
                    continue
                sentences, summaryFlags, _ = zip(*data)
                # print(len(summaryFlags))
                start = time.time()
                loss_chosen = []
                for num in range(len(summaryFlags[0])):
                    if summaryFlags[0][num] == 1:
                        loss_chosen.append(1)
                    else:
                        u = np.random.random()
                        if u < 0.12:
                            loss_chosen.append(1)
                        else:
                            loss_chosen.append(0)
                # print(loss_chosen)
                sentInds = self.words_to_inds(sentences[0], vocab)
                # print(sentences[0][0])
                sentInds, sentMask = self.pad_n_mask(sentInds)
                # print(sentInds)
                _, glnorm, prnorm, loss = self.optimize(sess, sentInds, summaryFlags[0], 
                sentMask, 1.0 - self.FLAGS.dropProb, self.FLAGS.learning_rate, loss_chosen) 
                batch_time = time.time() - start
                average_time = average_time * (i-1) / i + batch_time * 1/i
                eta = (self.FLAGS.train_size/4 - i) * average_time
                logging.info('Step %d / %d completed. Time taken = %f. Loss = %f. Global norm = %f. Parameter norm = %f. ETA = %f seconds until end of epoch' % (i, self.FLAGS.train_size/4, batch_time, loss, glnorm, prnorm, eta))
                # this print statement is nice to see things, but it is also slowing things down. may be a good idea to log 
                # instead, or print more rarely
                if i%2000 == 0: # check performance on dev set every once in a while, don't need to be full set actually, could 
                # separate 1000 from dev set or train set randomly and test on them
                # would also be nice to check how we are doing in terms of rouge every once in a while
                    totalDevLoss = 0.0
                    totalRouge = 0.0
                    totalAcc = 0.0
                    j = 0
                    for devData in self.readPreprocessedData(self.FLAGS.val_directory):
                        if len(devData) == 0:
                            continue
                        devSentences, devFlags, devSummary = zip(*devData)
                        chosen_inds = [1 for m in range(len(devFlags[0]))]
                        devInds = self.words_to_inds(devSentences[0], vocab)
                        devInds, devMask = self.pad_n_mask(devInds)
                        devLoss = self.test(sess,devInds, devFlags[0], devMask, chosen_inds)
                        totalDevLoss += devLoss[0]
                        rouge, accuracy = self.rouge_2(sess, devSentences[0], devInds, devFlags[0], devMask, devSummary[0])
                        totalRouge += rouge
                        totalAcc += accuracy
                        j += 1
                        if j%1000 == 0:
                            logging.info('Step %d / %d completed. Loss = %f.' % (j, self.FLAGS.dev_size, devLoss[0]))
                    logging.info('Total small-dev loss: %f. Total rouge: %f. Total accuracy: %f' % (totalDevLoss/100.0, totalRouge/100.0, totalAcc / 100.0))
            # Q: if we load models and continue training, make sure we update global_step to not overwrite on previously learned models
            # also the files saved after each epoch are about 700 mb, and numEpochs is set to 20 so make sure you have space if you plan to run
            # full training 
            self.saver.save(sess, self.FLAGS.save_directory + "/tldr1-epoch", global_step=e)
            averageRouge = 0.0
            averageAcc = 0.0
            j = 0
            for devData in self.readPreprocessedData(self.FLAGS.dev_directory): # at the end of epoch check performance on whole 
            # dev set, in terms of rouge
                if len(devData) == 0:
                    continue
                sentences, summaryFlags, refSummary = zip(*devData)
                # chosen_inds = [1 for m in range(len(devFlags[0])]
                sentInds = self.words_to_inds(sentences[0], vocab)
                sentInds, sentMask = self.pad_n_mask(sentInds)
                # rouge = self.rouge_2(sess, sentInds, summaryFlags[0], sentMask)
                # summary = self.generateSummary(sess, sentences[0], sentInds, summaryFlags[0], sentMask)
                # print(refSummary)
                # print("aykan")
                # print(sentences)
                # print(summaryFlags)
                # print(refSummary)
                rouge, accuracy = self.rouge_2(sess, sentences[0], sentInds, summaryFlags[0], sentMask, refSummary[0])
                # print(outs)
                # rouge, accuracy = zip(*outs)
                # print(len(summary))
                # print(sentences[0])
                # print(summary)
                averageRouge += rouge
                averageAcc += accuracy
                j += 1
                if j%1000 == 0:
                    logging.info('Step %d / %d completed. Rouge = %f.' % (j, self.FLAGS.dev_size, 
                    averageRouge / j))
            averageRouge = averageRouge / self.FLAGS.dev_size
            averageAcc = averageAcc / self.FLAGS.dev_size
            logging.info('Rouge score after epoch %d : %f' % (e, averageRouge))
            logging.info('Accuracy after epoch %d : %f' % (e, averageAcc))
# one way to speed things up a bit: decrease state_size flag to 100 in tldrTrain.py. this will also reduce accuracy
# another way: currently calling words_to_inds for each article, could instead save the inds version for each article in a txt file # and load those versions directly instead of actual words


# start with high learning rate, use decreasing learning rate
# make sure you don't run the code right before going to sleep, make sure that it finishes an epoch without any problems and saves without problem

# DONE except rouge       
# to do for tomorrow:
# softmax ce with logits
# setup_system, setup_embeddings, setup_loss
# train
# evaluate
# rouge
# variable scope and reuse
# dropout
# gradient clipping
# zero padding
# words to inds, vocab
# saver
# look at your own code if necessary
# initializer (xavier + global)
