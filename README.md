## For this project need directories and files:
- database - samples photo person (any photo people who will be recognition.)
- encoding_face - samples photo face who will be recognition. (only face for the best recognition)
- DB_Rocognition - this directory, will be create files of recognition faces
- deploy.prototxt.txt
- res10_300x300_ssd_iter_140000.caffemodel
- encodings.pickle - will be create when start file Face Encoding
- func_face.py - two function for detection and recognition faces.

1. Face prepare - this file prepare face pictures from the database directory and save it to the encoding_face directory.
2. Face Encoding -  this file to make database for working a siamese neural networks (make file encodings.pickle ) - encoding samples faces.
3. GetPhoto7 - get pictures from camera and save it to directory ./Camera/camera7 (where camera7 is name camera, I'm using 12 cameras) this file need working always.
4. Grouping images - this file grouping similar pictures of files to directory packages, this file need start every 5-15 minutes.
5. Face Recognition for Pandas - this file check group file and save recognition face people to directory DB_Rocognition