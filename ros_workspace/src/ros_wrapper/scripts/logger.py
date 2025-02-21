import rospy
import rosbag
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class TurtleBotLogger:
    def __init__(self):
        rospy.init_node('turtlebot_logger', anonymous=True)
        
        # Set bag file path
        self.bag = rosbag.Bag('turtlebot3_log.bag', 'w')

        # Subscribers
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.loginfo("TurtleBot Logger initialized. Recording data...")

    def imu_callback(self, data):
        self.bag.write('/imu', data)

    def odom_callback(self, data):
        self.bag.write('/odom', data)

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down and closing bag file.")
        finally:
            self.bag.close()

if __name__ == '__main__':
    logger = TurtleBotLogger()
    logger.run()
