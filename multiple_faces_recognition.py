#Tools
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import pickle

#CV2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from keras.models import load_model
import numpy as np
import tensorflow as tf

#Import the OpenCV and dlib libraries
import cv2
import dlib

#Multiple thread
import threading
import time

global graph
graph = tf.get_default_graph() 

# extract a single face from a given photograph
def extract_face(frame, person, required_size=(160, 160)):
	# extract the bounding box from the first face
	bounding_box = person['box']
	x1, y1, width, height = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
	#x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	
	# extract the face
	face = frame[y1:y2, x1:x2]
	
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	#yhat = model.predict(samples)
	with graph.as_default():
		yhat = model.predict(samples, batch_size=1, verbose=1)
		
	return yhat[0]

def make_embedding(face):
	# load face embeddings
	# convert each face in the test set to an embedding
	newTestX = list()
	embedding = get_embedding(model, face)
	newTestX.append(embedding)
	newTestX = asarray(newTestX)
	#print(newTestX.shape)

	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	testX = in_encoder.transform(newTestX)
	#print(testX)
	random_face_emb = testX[0]

	# prediction for the face
	# prediction for the face
	samples = expand_dims(random_face_emb, axis=0)
	yhat_prob = loaded_model.predict_proba(samples)

	y_pred_test_classes = np.argmax(yhat_prob, axis=1)
	y_pred_test_max_probas = np.max(yhat_prob, axis=1)

	# get name
	class_probability = y_pred_test_max_probas[0] * 100
	predict_names = out_encoder.inverse_transform(y_pred_test_classes)
	#print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	return predict_names, class_probability

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

# load the model from disk
loaded_model = pickle.load(open('svm_model.sav', 'rb'))

# load face embeddings
testy = ['nguyen_van_minh', 'tran_tuan_canh', 'tuyet_huong']
print("Test y label: ", testy)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(testy)
testy = out_encoder.transform(testy)


# create the detector, using default weights
detector = MTCNN()

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder
#faceCascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid, baseImage, person):
	# for person in faces:
	face = extract_face(baseImage, person)
	predict_names, class_probability = make_embedding(face)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

	time.sleep(2)
	faceNames[ fid ] = str(predict_names[0])




def detectAndTrackMultipleFaces():
	#Open the first webcame device
	capture = cv2.VideoCapture(0)

	#Create two opencv named windows
	#cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

	#Position the windows next to eachother
	#cv2.moveWindow("base-image",0,100)
	cv2.moveWindow("result-image",400,100)

	#Start the window thread for the two windows we are using
	cv2.startWindowThread()

	#The color of the rectangle we draw around the face
	rectangleColor = (0,165,255)

	#variables holding the current frame number and the current faceid
	frameCounter = 0
	currentFaceID = 0

	#Variables holding the correlation trackers and the name per faceid
	faceTrackers = {}
	faceNames = {}

	try:
		while True:
			#Retrieve the latest image from the webcam
			rc,baseImage = capture.read()

			#Resize the image to 320x240
			#baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))

			#Check if a key was pressed and if it was Q, then break
			#from the infinite loop
			pressedKey = cv2.waitKey(2)
			if pressedKey == ord('Q'):
				break



			#Result image is the image we will show the user, which is a
			#combination of the original image from the webcam and the
			#overlayed rectangle for the largest face
			resultImage = baseImage.copy()




			#STEPS:
			# * Update all trackers and remove the ones that are not 
			#   relevant anymore
			# * Every 10 frames:
			#       + Use face detection on the current frame and look
			#         for faces. 
			#       + For each found face, check if centerpoint is within
			#         existing tracked box. If so, nothing to do
			#       + If centerpoint is NOT in existing tracked box, then
			#         we add a new tracker with a new face-id


			#Increase the framecounter
			frameCounter += 1 



			#Update all the trackers and remove the ones for which the update
			#indicated the quality was not good enough
			fidsToDelete = []
			for fid in faceTrackers.keys():
				trackingQuality = faceTrackers[ fid ].update( baseImage )

				#If the tracking quality is good enough, we must delete
				#this tracker
				if trackingQuality < 7:
					fidsToDelete.append( fid )

			for fid in fidsToDelete:
				print("Removing fid " + str(fid) + " from list of trackers")
				faceTrackers.pop( fid , None )




			#Every 10 frames, we will have to determine which faces
			#are present in the frame
			if (frameCounter % 10) == 0:



				#For the face detection, we need to make use of a gray
				#colored image so we will convert the baseImage to a
				#gray-based image
				gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
				#Now use the haar cascade detector to find all faces
				#in the image
				#faces = faceCascade.detectMultiScale(gray, 1.3, 5)
				# detect faces in the image
				faces = detector.detect_faces(baseImage)
				#print("Faces: ", faces)

				#Loop over all faces and check if the area for this
				#face is the largest so far
				#We need to convert it to int here because of the
				#requirement of the dlib tracker. If we omit the cast to
				#int here, you will get cast errors since the detector
				#returns numpy.int32 and the tracker requires an int
				for person in faces:
					bounding_box = person['box']
					(startX, startY, endX, endY) = (bounding_box[0], bounding_box[1],
												bounding_box[2],bounding_box[3])
					x = int(startX)
					y = int(startY)
					w = int(endX)
					h = int(endY)


					#calculate the centerpoint
					x_bar = x + 0.5 * w
					y_bar = y + 0.5 * h



					#Variable holding information which faceid we 
					#matched with
					matchedFid = None

					#Now loop over all the trackers and check if the 
					#centerpoint of the face is within the box of a 
					#tracker
					for fid in faceTrackers.keys():
						tracked_position =  faceTrackers[fid].get_position()

						t_x = int(tracked_position.left())
						t_y = int(tracked_position.top())
						t_w = int(tracked_position.width())
						t_h = int(tracked_position.height())


						#calculate the centerpoint
						t_x_bar = t_x + 0.5 * t_w
						t_y_bar = t_y + 0.5 * t_h

						#check if the centerpoint of the face is within the 
						#rectangleof a tracker region. Also, the centerpoint
						#of the tracker region must be within the region 
						#detected as a face. If both of these conditions hold
						#we have a match
						if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
							 ( t_y <= y_bar   <= (t_y + t_h)) and 
							 ( x   <= t_x_bar <= (x   + w  )) and 
							 ( y   <= t_y_bar <= (y   + h  ))):
							matchedFid = fid


					#If no matched fid, then we have to create a new tracker
					if matchedFid is None:

						print("Creating new tracker " + str(currentFaceID))

						#Create and store the tracker 
						tracker = dlib.correlation_tracker()
						tracker.start_track(baseImage,
											dlib.rectangle( x,
															y,
															x+w,
															y+h))

						faceTrackers[ currentFaceID ] = tracker

						#Start a new thread that is used to simulate 
						#face recognition. This is not yet implemented in this
						#version :)
						t = threading.Thread( target = doRecognizePerson ,
											   args=(faceNames, currentFaceID, baseImage, person))
						t.start()
						#doRecognizePerson(faceNames, currentFaceID, baseImage, person)
						#Increase the currentFaceID counter
						currentFaceID += 1




			#Now loop over all the trackers we have and draw the rectangle
			#around the detected faces. If we 'know' the name for this person
			#(i.e. the recognition thread is finished), we print the name
			#of the person, otherwise the message indicating we are detecting
			#the name of the person
			for fid in faceTrackers.keys():
				tracked_position =  faceTrackers[fid].get_position()

				t_x = int(tracked_position.left())
				t_y = int(tracked_position.top())
				t_w = int(tracked_position.width())
				t_h = int(tracked_position.height())

				cv2.rectangle(resultImage, (t_x, t_y),
										(t_x + t_w , t_y + t_h),
										rectangleColor ,2)


				if fid in faceNames.keys():
					cv2.putText(resultImage, faceNames[fid] , 
								(int(t_x + t_w/2), int(t_y)), 
								cv2.FONT_HERSHEY_SIMPLEX,
								0.5, (255, 255, 255), 2)
				else:
					cv2.putText(resultImage, "Detecting..." , 
								(int(t_x + t_w/2), int(t_y)), 
								cv2.FONT_HERSHEY_SIMPLEX,
								0.5, (255, 255, 255), 2)






			#Since we want to show something larger on the screen than the
			#original 320x240, we resize the image again
			#
			#Note that it would also be possible to keep the large version
			#of the baseimage and make the result image a copy of this large
			#base image and use the scaling factor to draw the rectangle
			#at the right coordinates.
			largeResult = cv2.resize(resultImage,
									 (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

			#Finally, we want to show the images on the screen
			#cv2.imshow("base-image", baseImage)
			cv2.imshow("result-image", largeResult)


			if cv2.waitKey(20) & 0xFF == ord("q"):
				break

	#To ensure we can also deal with the user pressing Ctrl-C in the console
	#we have to check for the KeyboardInterrupt exception and break out of
	#the main loop
	except KeyboardInterrupt as e:
		pass

	#Destroy any OpenCV windows and exit the application
	cv2.destroyAllWindows()
	exit(0)


if __name__ == '__main__':
	detectAndTrackMultipleFaces()