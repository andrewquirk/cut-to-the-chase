import dataUtilPreprocess
import random
import ROUGEutil
from pythonrouge.pythonrouge import Pythonrouge

#For each article. Run the ROUGE algorithm on each sentence in the article to each sentence in the summary.
#For each sentence in the summary, the sentence in the article with the highest ROUGE score will be considered
#'important' and therefore in the summary and will given a label of 1. The rest will be considered not important
# and given a 0. 

#Represent the article as list of sentences
#Represent the inclusion as a list of labels (0 for not included and 1 for included)

#For each file, this program will create a processedFile where each line of the file is:
#sentence \t flag
#where sentece is a sentence from the article and flag is 1 if the sentence has content similar
#to the summary and 0 otherwise

ROUGE_dir = './pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
data_dir = './pythonrouge/RELEASE-1.5.5/data/'

def main():

	inputDirectory = raw_input('Directory of data? ')
	outputDirectory = raw_input('Directory to output files? ')

	dataReader = dataUtilPreprocess.storyDataReader(inputDirectory, 40)

	batchNum = 0
	articleNum = 0

	for batch in dataReader.readStoryData():

		print 'Reading batch {} and {} articles have been processed'.format(batchNum, articleNum)
		for fileName, data in batch.items():

			article = data[0] #list of sentences from the article
			summarySentences = data[1] #list of summary sentences

			#for each summary sentence, find the most relavent article sentence
			relaventSentences = set()
			for summarySentence in summarySentences:

				if len(summarySentence) == 0:
					continue

				maxScore = float('-inf')
				closestSentence = article[0]
				for articleSentence in article:

					if len(articleSentence) == 0:
						continue

					articleScore = computeRougeScore(articleSentence, summarySentence)

					if articleScore > maxScore:
						maxScore = articleScore
						closestSentence = articleSentence

				relaventSentences.add(closestSentence)

			articleNum += 1
			writeToOutput(outputDirectory +'/processed_' + fileName, article, relaventSentences)

		batchNum += 1

#Writes a processed file to output which indicates which sentences are relavent
# to the summary or not.
def writeToOutput(fileName, article, relaventSentences):
	with open(fileName, 'w') as f:

		for sentence in article:
			if len(sentence) == 0:
				continue
			
			if sentence in relaventSentences:
				flag = 1
			else:
				flag = 0
			f.write(sentence + '\t' + str(flag) + '\n')


#Computes the ROUGE score between a sentence from the article and a sentence from the summary
#and returns the result 
def computeRougeScore(articleSentence, summarySentence):
	#REFERNCE IN CASE WE WANT TO USE ROUGE LIKE THIS LATER
	# rouge = Pythonrouge(summary_file_exist=False,
 #                    summary=summarySentence, reference=articleSentence,
 #                    n_gram=1, ROUGE_SU4=False, ROUGE_L=False,
 #                    recall_only=True, stemming=True, stopwords=True,
 #                    word_level=True, length_limit=True, length=50,
 #                    use_cf=False, cf=95, scoring_formula='average',
 #                    resampling=False, samples=1000, favor=True, p=0.5)
	# score = rouge.calc_score()
	# return score['ROUGE-1']
#RUNS TOO SLOWLY ^^


	return ROUGEutil.rougeOne(articleSentence, summarySentence) + ROUGEutil.rougeTwo(articleSentence, summarySentence)


if __name__ == '__main__':
	main()