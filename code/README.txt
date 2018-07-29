
Cut to The Chase: An Extractive Approach to Machine Summarization

This project was developed for CS221 at Stanford University by Ian Hodge (CS), Andrew Quirk (CS), and Aykan Ozturk (EE) with the mentorship of CA Michael Xing.

The project’s goal was to provide a mechanism for automatic summarization of news articles.  We used the DailyMail dataset (https://cs.nyu.edu/~kcho/DMQA/) of approximate 200,000 news articles and given summaries.  We used the ROUGE metric to evaluate “successful” summaries, and our final weighted LSTM model achieved a ROUGE score of 6.8 with 85% accuracy, and our under sampled LSTM model achieved a ROUGE score of 6.9 with 85% accuracy.  Our full model and techniques are explained in this paper.

To put your own data in run splitData.py to put it into training, test, and dev sets.  Then, move those folders into the data/dailymail_processed folder.

The code.zip we uploaded includes an example of just 100 samples from each of the three (test,dev, and train) for brevity.

In all folders, run tldrTrain.py to run the model.  All have already been trained on four epochs and will continue from there.

If you are restoring one of the pre-trained models, make sure that the correct files are in ./model and the testing data is in ./data/dailymail_processed, then run tldrTrain.py.  Custom file paths can also be set in tldrTrain.py.  You can comment out the training in tldr.py if you want to skip directly to testing.


requirements.txt
This file makes sure you have the correct libraries installed.

flagsClass.py
This simple file just sets the flags to be referenced in the other files.

vocab.dat
This file holds all of the words that could be referenced by glove.

glove.trimmed.100.npz
This file holds the glove word embeddings to be read by other files.

splitData.py
Simple script that splits up the data into different folders for training, dev and testing.  You just need to make sure the path is correctly set up to the folder where your data is.

rougeOracle.py
Command: python rougeOracle.py
The program used to calculate our oracle (where we assumed 100% accuracy)

tldrTrain.py
This file contains the flags that are the hyper parameters that we used to tune the model.  We initially had state_size at 200 but reduced it to 100 to speed up training.  

If different training and testing sets are added, the train, dev, and test sizes would need to be edited.  
In addition, the learning rate could be lowered or raised to speed up training or increase accuracy.
The numEpochs is set to 20, but we were only able to train for four epochs due to time constraints.   Given more time, it would be interesting to train for the full length and observe results.

The other methods simply initialize the vocab and model.  If a checkpoint exists, the model tries to renew the parameters from the checkpoint.

tldr.py
This section constitutes the heart of the model, and does the bulk of the work of encoding, decoding, and classifying.  The train method is the one that implements training and testing the model.  In each epoch, it goes through the training data and updates the model.  Every epoch it saves to checkpoint, so the parameters can be restored from the previous epochs.  For purely testing purposes, the training section can be commented out and only the training set can be looped through.  If the checkpoint and meta data is loaded in ./model then it will automatically restore the parameters from the furthest epoch and use those to test.

./lstm_undersampled/tldr.py
This file is the same as tldr.py above, loading the undersampled LSTM model.

./simple_neural/tldr.py
This file is the same as tldr.py above, loading the undersampled LSTM model.

./lstm_weighted/tldr.py
This file is the same as tldr.py above, loading the weighted LSTM model.

preprocessGlove.py
Command: python preprocessGlove.py
Reads the vocab file created by createVocab and finds the corresponding GloVe vector for the corresponding word

createVocab.py
Command: python createVocab.py
Goes through all the articles and counts how many times a unique word appears so that we can create a vocabulary file.


ROUGEutil.py
Import this file to easily run a ROUGE-1 or the harmonic mean of the ROUGE-2 metric 

dataUtilPreprocess.py
Defines the storyDataReader class which allows to easily read in our data.
Reads each file in in the following format: filename -> (list of contentSentences, list of highlights) and returns a map from string to tuple
dataReaderPreprocessed.py
Import this file to declare an iterator that allows you to easily read in preprocessed data
The iterator reads in the sentences in the following tuple format:
	([sentence for sentence in article], [1 if article is included and 0 otherwise])

dataPreprocess.py
Command: python dataPreprocess.py
This file represents the bulk of the preprocessing code and algorithm. It reads in the articles and attempts to identify the most important sentences using the ROUGE-2 score.


