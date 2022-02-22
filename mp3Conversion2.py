import os
from os import path
from pydub import AudioSegment
import ffprobe
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import ffprobe

genres = ['Pop']

def convertAndMove(pathFrom, pathTo):
	sound = AudioSegment.from_mp3(pathFrom)
	sound.export(pathTo, format="wav")

def channelAndMove(pathFrom, pathTo):
	print(pathFrom)
	sound = AudioSegment.from_wav(pathFrom)
	sound = sound.set_channels(1)
	sound.export(pathTo, format="wav")

def doNothing():
	return 0

templateDirectory = '/Users/NJDStepinac/Desktop/SongReader2021/DS2/'

for genre in genres:
	originalFolderDirectory = templateDirectory + genre  
	newFolderDirectory = templateDirectory + genre + 'wav'                                                  
	for file in os.listdir(originalFolderDirectory):
		if (str(file) == '.DS_Store'):
			doNothing()
		else:
			#Get directories to music
			musicOriginalPath = originalFolderDirectory + "/" + file
			musicNewPath = newFolderDirectory + "/" + file[:-4] + '.wav'
			#Convert to Wav and save
			convertAndMove(musicOriginalPath, musicNewPath)

for genre in genres:
	originalFolderDirectory = templateDirectory + genre + 'wav'
	newFolderDirectory = templateDirectory + genre + 'wav1Ch'
	for file in os.listdir(originalFolderDirectory):
		if (str(file) == '.DS_Store'):
			doNothing()
		else:
			#Get directories to music
			musicOriginalPath = originalFolderDirectory + "/" + file
			musicNewPath = newFolderDirectory + "/" + file
			#Convert to Wav and save
			channelAndMove(musicOriginalPath, musicNewPath)





'''
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/audio_to_midi/__init__.py
'''