import os

#Reads in the the preprocessed data into the follwing format:
#([sentence for sentence in article], [1 if article is included and 0 otherwise])


def readPreprocessedData(dirName, batchSize):

	data = []

	for fileName in os.listdir(dirName):

		with open(fileName, 'r') as f:

			sentences = []
			summaryFlags = []
			for line in f:

				if len(line) == 0:
					continue

				parts = line.split()

				sentences.append(parts[0])
				summaryFlags.append(parts[1])

			data.append((sentences, summaryFlags))

		if len(data) == batchSize:
			yield data
			data = []

	yield data
				

