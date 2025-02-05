# Mini Project for Mobile Robot Navigation with AI

1. Train an AI model for state estimation
   * collect IMU data and ground truth data using ROS gazebo sim
   * implement an NN architecture to predict change in heading angle and displacement between readings
   * train and evaluate network (vary hyperparameters, including network architecture!); keep track of results
2. Train an AI model for robot control
   * train a DQN/DDPG architecture for robot control using the ROS gazebo framework using raw input data (i.e., more data points than example from class or add a camera and train based on raw camera image)
   * valuate performance of network (vary hyperparameters, including network architecture!); keep track of results
3. Train an AI model for YCB object classification
   * Train an object classification model using the provided dataset to classify the corresponding YCB objects
4. Provide a working script using the turtlebot gym environment that:
   * Navigates the turtlebot solely by exploiting the model trained in step
   * Includes a ROS wrapper for the AI classification model from step 3 that subscribes to the Turtlebot camera and classifies each image
   * Records relevant topics (imu, etc.) so that the AI model trained for state estimation from step 1 can be evaluated
5. Compile a presentation that includes:
   * Detailed methods and results for the individual AI models (1 – 3) using at least the following metrics
   * AI for state estimation:
     * Comparison to classical IMU propagation
   * AI for robot control:
     * Time without crashing
   * Time without crashing:
     * Precision and recall
