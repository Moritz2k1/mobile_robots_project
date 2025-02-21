#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# ------------------------------------------------
# ROS Node: AI Classifier Wrapper
# ------------------------------------------------
class ImageClassifierNode:
    def __init__(self):
        rospy.init_node('ai_classifier_node', anonymous=True)
        
        # Use GPU if available, else CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model.
        self.model = self.load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize CvBridge.
        self.bridge = CvBridge()
        
        # Define preprocessing (must match your training transforms).
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # adjust if needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Subscribe to the Turtlebot camera topic.
        self.image_sub = rospy.Subscriber(
            "/turtlebot3_burger/camera1/image_raw", Image, self.image_callback
        )
        rospy.loginfo("AI Classifier Node initialized and subscribed to camera topic.")
    
    def load_model(self):
        # Create the ResNet18 architecture.
        model = models.resnet18(weights=None)  # Do not load ImageNet weights.
        num_ftrs = model.fc.in_features
        
        # Set the number of classes to match the checkpoint (change to 3 if your checkpoint was trained for 3 classes).
        num_classes = 3
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load the trained model weights.
        default_path = "/home/moritz/ros_workspace/src/ai_classifier_node/model.pt"
        model_path = rospy.get_param("~model_path", default_path)
        model_path = os.path.expanduser(model_path)  # Expand '~'
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            rospy.loginfo("Model loaded successfully from %s", model_path)
        except Exception as e:
            rospy.logerr("Failed to load model: %s", e)
        return model
    
    def image_callback(self, img_msg):
        try:
            # Convert the ROS Image message to an OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return
        
        # Convert from BGR to RGB.
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image.
        input_image = self.transform(rgb_image)
        input_image = input_image.unsqueeze(0).to(self.device)  # add batch dimension
        
        # Run inference.
        with torch.no_grad():
            outputs = self.model(input_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        
        rospy.loginfo("Predicted class: %d", predicted_class)
        
if __name__ == '__main__':
    try:
        node = ImageClassifierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

