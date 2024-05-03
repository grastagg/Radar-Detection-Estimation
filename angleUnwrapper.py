from collections import deque
import numpy as np

class AngleUnwrapper:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.angle_buffer = deque(maxlen=window_size)

    def add_angle(self, angle):
        self.angle_buffer.append(angle)

    def unwrap_angles(self):
        unwrapped_angles = []
        prev_angle = None
        for angle in self.angle_buffer:
            if prev_angle is not None:
                # Compute the difference between the current angle and the previous angle
                diff = angle - prev_angle
                
                # Adjust the difference to be within the range [-pi, pi)
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                
                # Add the adjusted difference to the previous angle to get the unwrapped angle
                angle = prev_angle + diff
            
            # Store the unwrapped angle
            unwrapped_angles.append(angle)
            
            # Update the previous angle for the next iteration
            prev_angle = angle
            
        self.angle_buffer = deque(unwrapped_angles, maxlen=self.window_size)
        
        return unwrapped_angles