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

#Import functions
# from face_extract import extract_face, load_faces, load_dataset
# from face_embedding import get_embedding


# extract a single face from a given photograph
def extract_face(frame, results, required_size=(160, 160)):
	# # load image from file
	# image = Image.open(filename)
	
	# convert to RGB, if needed
	#image = image.convert('RGB')
	#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#print("image: ", image)
	
	# convert to array
	#pixels = asarray(image)
	#print("pixels: ", pixels)
	

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

# # load images and extract faces for all images in a directory
# def load_faces(directory):
# 	faces = list()
# 	# enumerate files
# 	for filename in listdir(directory):
# 		# path
# 		path = directory + filename
# 		# get face
# 		face = extract_face(path)
# 		# store
# 		faces.append(face)
# 	return faces

# # load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset(directory):
# 	X = list()
# 	# enumerate folders, on per class
# 	for subdir in listdir(directory):
# 		# path
# 		path = directory + subdir + '/'
# 		print(path)
# 		# skip any files that might be in the dir
# 		if not isdir(path):
# 			continue
# 		# load all faces in the subdirectory
# 		faces = load_faces(path)
# 		# create labels
# 		labels = [subdir for _ in range(len(faces))]
# 		# summarize progress
# 		print('>loaded %d examples for class: %s' % (len(faces), subdir))
# 		# store
# 		X.extend(faces)
# 		#y.extend(labels)
# 	return asarray(X)#, asarray(y)

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

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

# load the model from disk
loaded_model = pickle.load(open('svm_model.sav', 'rb'))

# load face embeddings
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

video_src = 'ben1.mp4'
cap = cv2.VideoCapture(video_src)
while True: 
	#Capture frame-by-frame
	ret, frame = cap.read()
	#print(frame)
	# load the photo and extract the face
	#testX_faces, testy_faces = load_dataset('./images/')

		# create the detector, using default weights
	detector = MTCNN()
	
	# detect faces in the image
	results = detector.detect_faces(frame)
	#print("results: ", results)
	
	if results != []:
		for person in results:
			bounding_box = person['box']
			keypoints = person['keypoints']
	
			cv2.rectangle(frame,
						  (bounding_box[0], bounding_box[1]),
						  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
						  (0,155,255),
						  2)
	
			# cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
			# cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
			# cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
			# cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
			# cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)


			face = extract_face(frame, results)
			faces = list()
			faces.append(face)
			testX_faces = list()
			testX_faces.extend(faces)
			#print(testX_faces, testy_faces)


			# load face embeddings
			# convert each face in the test set to an embedding
			newTestX = list()
			for face_pixels in testX_faces:
				embedding = get_embedding(model, face_pixels)
				newTestX.append(embedding)
			newTestX = asarray(newTestX)
			#print(newTestX.shape)

			# normalize input vectors
			in_encoder = Normalizer(norm='l2')
			testX = in_encoder.transform(newTestX)

			# label encode targets
			out_encoder = LabelEncoder()
			out_encoder.fit(testy)
			testy = out_encoder.transform(testy)

			# test model on a random example fromq the test dataset
			selection = choice([i for i in range(newTestX.shape[0])])
			random_face_pixels = testX_faces[selection]
			random_face_emb = testX[selection]
			random_face_class = testy[selection]
			random_face_name = out_encoder.inverse_transform([random_face_class])


			

			# prediction for the face
			samples = expand_dims(random_face_emb, axis=0)
			yhat_class = loaded_model.predict(samples)
			yhat_prob = loaded_model.predict_proba(samples)

			# get name
			class_index = yhat_class[0]
			class_probability = yhat_prob[0,class_index] * 100
			predict_names = out_encoder.inverse_transform(yhat_class)
			print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
			print('Expected: %s' % random_face_name[0])

			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (255,255,255)
			stroke = 2

			cv2.putText(frame, str(predict_names[0]), (200, 200), font, 1, color, stroke, cv2.LINE_AA)

	cv2.imshow('video', frame)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		break
	# # plot for fun
	# pyplot.imshow(random_face_pixels)
	# title = '%s (%.3f)' % (predict_names[0], class_probability)
	# pyplot.title(title)
	# pyplot.show()