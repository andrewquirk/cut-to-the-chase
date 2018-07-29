import os

HIGHLIGHT_FLAG = '@highlight'
NBSP = '\xc2\xa0'

#Reads each file in in the following format:
#filename -> (list of contentSentences, list of highlights)
#returns a map from string to tuple
class storyDataReader(object):

	#Pass in the name of the directory and the desired batch size
	def __init__ (self, dirName, batchSize):

		self.dirName = dirName
		self.batchSize = batchSize

	#Returns an iterator over the directory, one can iterator over all of the data with something such as:
	#for batch in readStoryData:
	#	for data in batch:
	#		process(data)
	def readStoryData(self):
		storyDirectory = self.dirName

		data = {}
		for fileName in os.listdir(storyDirectory):

			content = []
			f = open(storyDirectory + '/' + fileName, 'r')

			line = f.readline().strip()
			while HIGHLIGHT_FLAG not in line:
				content.append(line.replace(NBSP, ' '))			
				line = f.readline().strip()
			
			highlights = []
			#iternate until we hit the EOF, skips the white spaces in between highlights
			while(f.readline()):
				line = f.readline().strip()
				if HIGHLIGHT_FLAG not in line:
					highlights.append(line.replace(NBSP, ' '))

			contentString = ''.join(content)
			highlightString = '. '.join(highlights)

			data[fileName] = (content, highlights)
			f.close()

			if len(data) == self.batchSize:
				yield data
				data = {}

		yield data


 






