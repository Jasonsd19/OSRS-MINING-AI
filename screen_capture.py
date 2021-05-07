import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
from mss import mss
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pyautogui
import threading

# Path to the tensorflow model
weights_path = './checkpoints/ores-416'
# Size of image that the model accepts (we will resize the image to this size)
image_size = 416
# If True then openCV will display the real-time object detection, otherwise nothing will be displayed
show_video = True
# Intersection over Union threshold, the ratio of area overlap between the 'true box' and predicted box
iou_thresh = 0.45
# Score threshold, model will not return predictions with a score lower than what is specified below
score_thresh = 0.80

def main():
    # Configuration and initialization
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = image_size

    # Load our tensorflow model
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # Initialize events and threads
    e = threading.Event()
    t = threading.Thread(target=print, args=('Primed and Ready!',))
    t.start()

    static_inv_iron = 0
    with mss() as sct:
    # Area of screen we want to capture - feel free to change
        monitor = {"top": 0, "left": 0, "width": 1919, "height": 1079}
        while True:
            # Used in FPS calculations below, uncomment if wanted
            # start_time = time.time()
            
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold= iou_thresh,
                score_threshold= score_thresh,
            )

            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            # Image is the screen capture with boxes overlayed, iron_ores is a list containing the (x,y) coords
            # of all identified iron_ore, and inv_iron is a list containing the (x,y) coords of all identified
            # iron ore in the inventory
            image, iron_ores, inv_iron = utils.draw_bbox(frame, pred_bbox)

            # This code was used to mine the nearest ore, not needed anymore
            # center_pos = (int(monitor["width"] / 2), int(monitor["height"] / 2))
            # iron_abs_diff, iron_pos = closest_ore(iron_ores, center_pos)

            # Update the amount of iron ore in our inventory
            # every update triggers the event letting our
            # mining thread know we successfully mined an ore.
            if len(inv_iron) > static_inv_iron:
                static_inv_iron = len(inv_iron)
                e.set()
            if len(inv_iron) == 0:
                static_inv_iron = 0
            
            # If we have stopped mining, then mine another rock
            if not t.is_alive():
                e.clear()
                t = threading.Thread(target=mine_ore, args=(iron_ores[0], inv_iron, e,))
                t.start()

            # Used in FPS calculations, uncomment if wanted
            #fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)

            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            # Whether or not we display the screen capture
            if show_video:
                cv2.imshow("result", result)

            # Exit out of cv2 screen capture by pressing q will screen capture is active window
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

def mine_ore(pos, inv_iron, event):
# Function called by a seperate thread, empties inventory if >= 5 ores present
# and begins to mine the ore specified at pos, communicates with main thread through event.
# We return out of function if the event is triggered or 15 seconds pass.
    if len(inv_iron) >= 5:
        for iron_pos in inv_iron:
            pyautogui.moveTo(iron_pos[0], iron_pos[1])
            time.sleep(0.25)
            pyautogui.click()
    pyautogui.moveTo(pos[0], pos[1])
    time.sleep(0.25)
    pyautogui.click()
    event.wait(15)
    return

# def closest_ore(ores, pos): 
# # This function takes the squared difference between the player's position and the nearest ore's position
# # and returns the smallest difference (i.e. the nearest ore), along with it's position (x,y).
#     best_diff = 1000000000
#     ore_pos = (0,0)
#     for ore in ores:
#         x = (ore[0] - pos[0])**2
#         y = (ore[1] - pos[1])**2
#         diff = x + y
#         if diff < best_diff:
#             best_diff = diff
#             ore_pos = ore
#     return best_diff, ore_pos

if __name__ == '__main__':
    main()
