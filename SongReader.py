import os
import shutil
from tinytag import TinyTag

#Create all the directories for the songs
def makeDirectories():
	count = 0
	for firstDigit in range(0,2):
		for secondDigit in range (0, 10):
			for thirdDigit in range (0,10):
				directories.append('/Users/NJDStepinac/Desktop/Research Paper/fma_small/fma_small/' + str(firstDigit) + str(secondDigit) + str(thirdDigit))
				count += 1
				if (count == 156):
					return count


#Assign directories
directories = []
print(makeDirectories())

genres = {}

#Go through the directories
for directory in directories:
	checkCondition = False
	for fileName in os.listdir(directory):
		filePath = directory + '/' + fileName
		readFile = TinyTag.get(filePath)
		genre = readFile.genre
		if genre not in genres:
			genres[genre] = 0
		genres[genre] += 1



for genre in genres:
	if genre is not None:
		print(genre + ':' + str(genres[genre]) + '\n')