import cv2

video_path = 'C:\\Myfolder\\Code\\Python\\week4-5\\video\\pednet.mp4'

vidcap = cv2.VideoCapture(video_path)
success,frame = vidcap.read()
count = 0
while success:
  cv2.imwrite("week4-5\\frame_pednet\\frame%d.jpg" % count, frame)     # save frame as JPEG file      
  success,frame = vidcap.read()
  print('Read a new frame%d: ' % count, success)
  count += 1