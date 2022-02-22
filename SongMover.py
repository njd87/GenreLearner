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

def moveSong(genreSTR, path, file):
	if (genreSTR == 'Pop'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Pop")
		'''
	elif (genreSTR == 'Folk' or genreSTR == 'Free Folk'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Folk")
	elif (genreSTR == 'Rock' or genreSTR == 'Krautrock'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Rock")
	elif (genreSTR == 'Hip-Hop' or genreSTR == 'Hip Hop' or genreSTR == 'Hip-Hop Beats' or genreSTR == 'Rap'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Hip-Hop")
	elif (genreSTR == 'Electroacoustic' or genreSTR == 'Electronic' or genreSTR == 'Dubstep' or genreSTR == 'House' or genreSTR == 'electronic'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Electronic")
	elif (genreSTR == 'International' or genreSTR == 'Latin' or genreSTR == 'French' or genreSTR == 'Latin America' or genreSTR == 'Misc. International' or genreSTR == 'Middle East' or genreSTR == 'Ethnic' or genreSTR == 'Indian' or genreSTR == 'World'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Ethnic")
	elif (genreSTR == 'Soundtrack' or genreSTR == 'Cinematic'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Cinematic")
	elif (genreSTR == 'Experimental' or genreSTR == 'Progressive' or genreSTR == 'Experimental Pop' or genreSTR == 'Indie' or genreSTR == 'Indie -- Rock'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Experimental")
	elif (genreSTR == 'Reggae - Dub' or genreSTR == 'Lo-fi' or genreSTR == 'Jazz' or genreSTR == 'Free-Jazz' or genreSTR == 'Lo-Fi' or genreSTR == 'Acoustic' or genreSTR == 'Acoustic, Improvisation' or genreSTR == 'acoustic' or genreSTR == 'Acoustic,Instrumental,Ambient' or genreSTR == 'Instrumental'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Open")
	elif (genreSTR == 'Noise' or genreSTR == 'Noise-Rock'):
		shutil.move(path, "/Users/NJDStepinac/Desktop/SongReader2021/Songs/Noise")
		'''


#Assign directories
directories = []
print(makeDirectories())

genres = {}

#Go through the directories
for directory in directories:
	#get song name
	for fileName in os.listdir(directory):
		filePath = directory + '/' + fileName
		readFile = TinyTag.get(filePath)
		genre = readFile.genre
		genre = str(genre)
		moveSong(genre, filePath, fileName)




