# Step 0: Preparations
import tobii_research as tr
import time

# Step 1: Find the eye tracker
found_eyetrackers = tr.find_all_eyetrackers()

if not found_eyetrackers:
    print("No eye tracker found. Please connect one and try again.")
    exit()

# Use the first found eye tracker
my_eyetracker = found_eyetrackers[0]

print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)


# Step 3: Define callback function for gaze data
def gaze_data_callback(gaze_data):
    # Print gaze points of left and right eye
    print("Left eye: {0} \t Right eye: {1}".format(
        gaze_data['left_gaze_point_on_display_area'],
        gaze_data['right_gaze_point_on_display_area']))


# Subscribe to gaze data
print("\nStarting gaze data collection for 5 seconds...")
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

# Let it collect gaze data for 5 seconds
time.sleep(5)

# Step 4: Unsubscribe from gaze data
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
print("Stopped gaze data collection.")
