from __future__ import print_function
import time

def cd(seconds):
    while seconds > 0:
        rem_m = seconds // 60
        rem_s = seconds % 60
        remaining_time = str(rem_m) + " minutes " + str(rem_s) + " seconds"
        print(remaining_time, end='\r')
        seconds = seconds - 1
        time.sleep(1)
    
    print("Xxxxx xxxxX")

cd(60)
