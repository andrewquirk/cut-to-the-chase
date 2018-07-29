

#Performs the ROUGE-1 algorithm on two input strings
def rougeOne(reference, summary):

	count = 0
	for word in summary:
		if word in reference:
			count += 1

	return float(count)/len(summary)


#Performs the ROUGE-2 algorithm on two input strings
def rougeTwo(reference, summary):

	 bigramsReference = zip(reference, reference[1:])
	 bigramsSummary = zip(summary, summary[1:])

	 if len(bigramsSummary) == 0:
	 	return 0
	 
	 count = 0
	 for bigram in bigramsSummary:
	 	if bigram in bigramsReference:
	 		count += 1

	 return float(count)/len(bigramsSummary)