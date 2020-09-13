# Remember to install necessary libaries
tensorflow-gpu version 1.15
CUDA 10.2
opencv version 4.1.2

# How to run Attendace-Cheking-using-Facenet-and-MTCNN
Step 1: Run face_extract.py to extract the bounding box which cover human faces.
Step 2: Run face_embedding.py to extract (128d vector) the features of faces.
Step 3: Run face_classification to train SVM classifier to recognize the face of specific person.
Step 4: Run real_time_cv2.py to test the result on video or real-time camera.

