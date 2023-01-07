from deepface import DeepFace

DeepFace.stream('/Users/jaygopal/Documents/GitHub/FaceEmotionDetection/image_database', model_name='Facenet', detector_backend='mtcnn', time_threshold=3, frame_threshold=2)

