import dataUtilPreprocess
import random
import ROUGEutil
import os

FILE_NUM_TO_READ = 500

def main():

	inputDirectory = raw_input('Directory of data? ')
	

	maxRouge, minRouge, averageRouge = findRougeScores(inputDirectory)
	
	print 'Max Score = {}'.format(maxRouge)
	print 'Min Score = {}'.format(minRouge)
	print 'Average Score = {}'.format(averageRouge)


def findRougeScores(storyDirectory):

	fileCount = 0
	rougeScores = []
	for fileName in os.listdir(storyDirectory):

		if fileCount > FILE_NUM_TO_READ:
			return max(rougeScores), min(rougeScores), (float(sum(rougeScores))/len(rougeScores))

		f = open(storyDirectory + '/' + fileName, 'r')

		line = f.readline()

		pickedSentences = []
		while '@summary' not in line:
			words = line.split()
			flag = words[len(words) - 1]
			if flag == '1':
				pickedSentences.append(' '.join(words[:len(words) - 1:]))
			line = f.readline()

		rougeScores.append(ROUGEutil.rougeTwo(' '.join(pickedSentences), f.readline()))
		fileCount += 1
		f.close()





if __name__ == '__main__':
	main()