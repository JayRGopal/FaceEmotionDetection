from opendbm import FacialActivity

#make sure Docker is active to access the model
model = FacialActivity()
path = "sample.mp4"
model.fit(path)
landmark = model.get_landmark()


