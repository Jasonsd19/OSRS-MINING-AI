# OSRS-MINING-AI
NOTE: Most of the files in here are modified from https://github.com/theAIGuysCode/tensorflow-yolov4-tflite so all credit to the original creator. If you wish to learn how to convert yolov4 models to tensorflow and much more follow the link.

Here I trained a neural network using yolov4. The trained weights file was then used to create a tensorflow model. I then used the tensorflow model alongside openCV and some I/O python libraries such as pyautogui to automate the mining skill in the game Old School Runescape. This was a quick project and I learned a lot and also made many mistakes, however I am pleased with how the bot turned out. All of the files I have here are configured specifically for my trained neural network, the screen_capture script can definitely be modified and used for other models as well.

The darknet files contain the weights I've trained at 1400 iterations and 300 iterations, along with the custom dataset that I trained the model on as well, and various other files such as .names and .data files.
