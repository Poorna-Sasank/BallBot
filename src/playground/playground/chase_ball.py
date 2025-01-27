import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import threading

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # Subscriber for the camera image
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1  # Queue size of 1
        )

        # Publisher for robot velocity commands
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Variable to store the latest frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()  # Lock for thread safety
        
        # Flag to control the display loop
        self.running = True

        # Start a separate thread for spinning (to ensure image_callback keeps receiving new frames)
        self.spin_thread = threading.Thread(target=self.spin_thread_func)
        self.spin_thread.start()

        # OpenCV window setup
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800, 600)

        # Parameters for stopping distance
        self.min_ball_radius = 30   # Minimum radius of the ball (in pixels) to stop the bot
        self.max_ball_radius = 100  # Maximum radius of the ball (in pixels) for safe tracking

    def spin_thread_func(self):
        """Separate thread function for rclpy spinning."""
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):
        """Callback to receive and store the latest frame."""
        try:
            with self.frame_lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def display_image(self):
        """Main loop to process and display the latest frame."""
        while rclpy.ok() and self.running:
            # Check if there is a new frame available
            if self.latest_frame is not None:
                with self.frame_lock:
                    frame = self.latest_frame.copy()
                    self.latest_frame = None  # Clear the frame after processing

                # Process the current image
                red_mask, contour_mask, crosshair_mask = self.process_image(frame)

                # Add processed images as small overlays
                result = self.add_small_pictures(frame, [red_mask, contour_mask, crosshair_mask])

                # Show the latest frame
                cv2.imshow("frame", result)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        # Close OpenCV window after quitting
        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, img):
        """Process the image to detect the red ball and generate control commands."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize blank masks
        contour_mask = np.zeros_like(img)
        crosshair_mask = np.zeros_like(img)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                radius = int(np.sqrt(cv2.contourArea(largest_contour) / np.pi))
            else:
                cx, cy, radius = 0, 0, 0

            # Draw contour and centroid
            contour_mask = cv2.drawContours(np.zeros_like(img), [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(contour_mask, (cx, cy), 5, (0, 255, 0), -1)

            # Draw crosshair
            rows, cols = img.shape[:2]
            crosshair_mask = cv2.line(np.zeros_like(img), (cx, 0), (cx, rows), (0, 0, 255), 2)
            crosshair_mask = cv2.line(crosshair_mask, (0, cy), (cols, cy), (0, 0, 255), 2)
            crosshair_mask = cv2.line(crosshair_mask, (int(cols / 2), 0), (int(cols / 2), rows), (255, 0, 0), 2)
            
            if abs(cols / 2 - cx) > 20:
                if cols / 2 > cx:
                    msg.angular.z = 0.2  # Turn left
                else:
                    msg.angular.z = -0.2  # Turn right
            else:
                msg.linear.x = 0.2  # Move forward

            # # Check if the ball is too close (radius exceeds threshold)
            # if radius > self.min_ball_radius:
            #     # Stop the bot if the ball is too close
            #     msg.linear.x = 0.0
            #     msg.angular.z = 0.0
            # else:
            #     # Chase the ball if it's within a safe distance
            #     if abs(cols / 2 - cx) > 20:
            #         if cols / 2 > cx:
            #             msg.angular.z = 0.2  # Turn left
            #         else:
            #             msg.angular.z = -0.2  # Turn right
            #     else:
            #         msg.linear.x = 0.2  # Move forward

        # Publish the velocity command
        self.publisher.publish(msg)

        return red_mask, contour_mask, crosshair_mask

    def add_small_pictures(self, img, small_images, size=(160, 120)):
        """Add small images to the top row of the main image."""
        x_base_offset = 40
        y_base_offset = 10

        x_offset = x_base_offset
        y_offset = y_base_offset

        for small in small_images:
            small = cv2.resize(small, size)
            if len(small.shape) == 2:
                small = np.dstack((small, small, small))

            img[y_offset: y_offset + size[1], x_offset: x_offset + size[0]] = small
            x_offset += size[0] + x_base_offset

        return img

    def stop(self):
        """Stop the node and the spin thread."""
        self.running = False
        self.spin_thread.join()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        node.display_image()  # Run the display loop
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()  # Ensure the spin thread and node stop properly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()