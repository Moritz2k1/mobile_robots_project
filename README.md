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

## creating the ros workspace
to create the ros workspace
```bash
$ mkdir ros_workspace
$ cd ros_workspace
$ mkdir src
$ catkin_make
$ source devel/setup.bash
```

to create the package
```bash
$ cd src
$ catkin_create_pkg state_estimation_logger rospy std_msgs sensor_msgs nav_msgs geometry_msgs
$ cd state_estimation_logger
$ mkdir scripts
```
copy the wrapper file that should be used in ROS with all the other dependecies (in my case the logger.py file is sufficient) in the scripts folder

update CMakeLists.txt in my_turtlebot_logger by adding this line
```bash
catkin_install_python(PROGRAMS
  scripts/logger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

Add dependencies to the package.xml (if needed)
```bash
<depend>rospy</depend>
<depend>std_msgs</depend>
<depend>sensor_msgs</depend>
<depend>nav_msgs</depend>
<depend>geometry_msgs</depend>
```

create a launch file for the package 
```bash
$ mkdir launch
```
create a logger.launch file an put those lines inside
```bash
<launch>
    <!-- Launch TurtleBot3 simulation -->
    <include file="$(find turtlebot3_gazebo)/launch/turtlebot3_world.launch"/>

    <!-- Launch the custom logging node -->
    <node name="logger" pkg="state_estimation_logger" type="logger.py" output="screen"/>
</launch>
```

execute catkin_make again from the root of the ros workspace
then use the command:
```bash
$ roslaunch state_estimation_logger logger.launch
```
to launch the simulation