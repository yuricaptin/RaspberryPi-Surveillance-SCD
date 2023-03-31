# RaspberryPi-Sureveillance-SCD

Code for the project is entirely built off of python. Purpose is for the first pi to recognise the presented face. Then determines if the face is on the system and should be allowed to enter, if it isnt on the system then the person is denied entry.

The second raspberry Pi is responsible for the Apache web Server. The framework we will be using is the Django framework. This pi will recieve live camera footage from the first raspberry pi and display it on the website allowing the homeowner/admin to vieew it from anywhere.

The steps should be followed first in the RealProject.py
After that you will then follow in this sequence: 
Run RPTraining, RPEvaluation then RPRTT.

Mostly Just because of very little knowledge with how this works. 

Later on I will add in the django code in as well to live stream the camera from an ip address to the other raspberry pi.

Austin and James: The Images that I will deposit into the google drive will have the images of me and austin as well as
a collection of faces from the wild as well. Ill need to go back later on and do some data augmentation to help with the accuracy of the model.
