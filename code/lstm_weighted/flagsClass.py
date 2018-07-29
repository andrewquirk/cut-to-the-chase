class flags(object):
    def __init__(self, train_directory, dev_directory, glove_path, state_size, learning_rate, 
        maxSentenceLength, glove_dim, numClasses, save_directory, numEpochs, dropProb,
        trainSize, devSize, clipGradients):
        self.train_directory = train_directory
        self.dev_directory = dev_directory
        self.vocab_directory = vocab_directory
        self.glove_path = glove_path
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.maxSentenceLength = maxSentenceLength
        self.glove_dim = glove_dim
        self.numClasses = numClasses
        self.save_directory = save_directory
        self.numEpochs = numEpochs
        self.dropProb = dropProb
        self.trainSize = trainSize
        self.devSize = devSize
        self.clipGradients = clipGradients
