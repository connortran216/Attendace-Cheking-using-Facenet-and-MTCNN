# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import pickle

#CV2
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from os import listdir
from os.path import isdir
import matplotlib.pyplot as plt
from keras.models import load_model
import dlib
# #Import functions
# from sort import *


# extract a single face from a given photograph
def extract_face(frame, results, required_size=(160, 160)):
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	
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
	yhat = model.predict(samples)
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
		samples = expand_dims(random_face_emb, axis=0)
		yhat_class = loaded_model.predict(samples)
		yhat_prob = loaded_model.predict_proba(samples)

		# get name
		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		#print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
		return predict_names, class_probability

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

# load the model from disk
loaded_model = pickle.load(open('svm_model.sav', 'rb'))


testy = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
print("Test t label: ", testy)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(testy)
testy = out_encoder.transform(testy)


# create the detector, using default weights
detector = MTCNN()

frame_count = 0
trackers = []

video_src = 'ben1.mp4'
cap = cv2.VideoCapture(video_src)
while True: 
	#Capture frame-by-frame
	ret, frame = cap.read()
	#print(frame)
	if frame_count == 0: 
		# detect faces in the image
		results = detector.detect_faces(frame)
		#print("results: ", results)

		if results != []:
			#print("Box: ", results[0]["box"])
			#print("Confidence: ", results[0]["confidence"])
			for person in results:
				bounding_box = person['box']
				keypoints = person['keypoints']
		
				cv2.rectangle(frame,
							  (bounding_box[0], bounding_box[1]),
							  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
							  (0,155,255),
							  2)

			(startX, startY, endX, endY) = (bounding_box[0], bounding_box[1],
											bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3])
			# We need to initialize the tracker on the first frame
			# Create the correlation tracker - the object needs to be initialized
			# before it can be used
			tracker = dlib.correlation_tracker()

			# Start a track on face detected on first frame.
			rect = dlib.rectangle(bounding_box[0], bounding_box[1],
								  bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3])
			tracker.start_track(frame, rect)

			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
			trackers.append(tracker)
			print("HELLO")

			
	else:

		face = extract_face(frame, results)
		predict_names, class_probability = make_embedding(face)

			
		# Else we just attempt to track from the previous frame
		# track all the detected faces
		for tracker in trackers:
			tracker.update(frame)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# draw bounding box
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

		font = cv2.FONT_HERSHEY_SIMPLEX
		color = (255,255,255)
		stroke = 2
		if class_probability > 50:
			cv2.putText(frame, str(predict_names[0]) + ": " + str(class_probability), 
						(startX, startY), font, 1, color, stroke, cv2.LINE_AA)
		
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		cv2.putText(frame, "FPS: " + str(fps), (50,50), font, 1, color, stroke, cv2.LINE_AA)

	frame_count += 1
	print("frame: ", frame_count)

	cv2.imshow('video', frame)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		break
