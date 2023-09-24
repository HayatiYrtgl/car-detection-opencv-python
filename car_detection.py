import cv2 
import numpy as np
from PIL import Image as img
from PIL import ImageTk 
import tkinter as tk


# read the video
video = cv2.VideoCapture("images/car_detection.mp4")

# create the app
app = tk.Tk()

# app features
app.geometry("900x900+150+10")
app.title("car detection")

# create countinglabel
count_label = tk.Label(app, text="frame number : 0")
count_label.pack(padx=10, pady=10)

# image label
image_label = tk.Label(app)
image_label.pack()

# create backgroundsubtractor
bg_masker = cv2.createBackgroundSubtractorMOG2(detectShadows=False) 

# sayi
sayi = 0

# car number
car_number = 0

# function to show
def show_frames():
	global sayi
	global car_number	
	# read frame from the video
	ret, frame = video.read()
	
	# image processing
	frame = cv2.resize(frame, (900, 900))
	
	# roi
	roi = frame[500:900, 200:800]
	
	# blur the image
	blurred = cv2.blur(roi, (5, 5))
	
	# with morph open  remove remove noise
	morphed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, (5, 5))
	morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, (5, 5))
	
	# bg subtractor
	frame_processed = bg_masker.apply(morphed)
	
	# frame counter
	sayi += 1
	
	if sayi > 700:
		
		# draw contours
		contours, hier = cv2.findContours(frame_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# draw contours
		for cnt in contours:
			
			# if area is qreater than 1000
			if cv2.contourArea(cnt) > 4000:
				
				 # draw the center
				 m = cv2.moments(cnt)
				 
				 y = int(m["m01"]/m["m00"])
				 
				 x = int(m["m10"]/m["m00"]) 
				 
				 cv2.circle(frame[500:900, 200:800], (x,y), 5, (0, 255, 0), -1)
				 
	
	# cv2 to PIL image
	pil_image = img.fromarray(frame)
	
	# photo image
	photo_image = ImageTk.PhotoImage(image=pil_image)
	
	# from pil to tkinter
	image_label.photo_image = photo_image
	
	# configuration
	image_label.configure(image=photo_image)
	
	# loop
	image_label.after(10, show_frames)
	
	count_label.configure(text="frame number: "+str(sayi))

# start the show frrame function
show_frames()

# bind the app
app.bind("<Escape>", lambda x: app.quit())
app.mainloop()