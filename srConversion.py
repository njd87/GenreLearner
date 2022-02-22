import os
from os import path
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import librosa
import soundfile

genres = ['Cinematic', 'Electronic', 'Ethnic', 'Experimental', 'Folk', 'Hip-Hop', 'Noise', 'Open', 'Pop', 'Rock']

def convertAndMove(pathFrom, pathTo):
	print(pathTo)
	y, s = librosa.load(pathFrom, sr=8000)
	soundfile.write(pathTo, y, 8000)

'''
def channelAndMove(pathFrom, pathTo):
	print(pathFrom)
	sound = AudioSegment.from_wav(pathFrom)
	sound = sound.set_channels(1)
	sound.export(pathTo, format="wav")
'''


def doNothing():
	return 0

originFolder = "/SongWavs1Ch/"
destinationFolder = "/SongWavs1Ch16/"

templateDirectory = '/Users/NJDStepinac/Desktop/SongReader2021'

for genre in genres:
	originalFolderDirectory = templateDirectory + "/SongWavs1Ch/" + genre  
	newFolderDirectory = templateDirectory + "/SongWavs1Ch16/" + genre                                                    
	for file in os.listdir(originalFolderDirectory):
		#Get directories to music
		musicOriginalPath = originalFolderDirectory + "/" + file
		musicNewPath = newFolderDirectory + "/" + file[:-4] + '.wav'
		#Convert to Wav and save
		convertAndMove(musicOriginalPath, musicNewPath)



'''
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/audio_to_midi/__init__.py
'''