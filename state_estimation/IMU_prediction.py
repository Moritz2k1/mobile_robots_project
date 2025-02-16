import rosbag
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num
from datetime import datetime
from model import LSTM
import argparse
from scipy.interpolate import interp1d
from torch.autograd import Variable

def extract_data(bag_file_path):
    time_stamps = []
    x_values = []
    y_values = []
    heading_values = []
    imu_msgs = []

    # Open the bag file
    with rosbag.Bag(bag_file_path) as bag:
        for topic, msg, t in bag.read_messages(topics=['/odom', '/imu']):
            if topic == '/odom':
                time_stamps.append(datetime.fromtimestamp(t.to_sec()))
                x_values.append(msg.pose.pose.position.x)
                y_values.append(msg.pose.pose.position.y)
                orientation = msg.pose.pose.orientation
                yaw = 2 * np.arctan2(orientation.z, orientation.w)  # Convert quaternion to yaw
                heading_values.append(yaw)
            elif topic == '/imu':
                imu_msgs.append((t, msg))

    return time_stamps, x_values, y_values, heading_values, imu_msgs


def compute_data(starting_position, imu_msgs, odom_timestamps):
    x_values_IMU = []
    y_values_IMU = []
    heading_values_IMU = []

    current_x = starting_position['x']
    current_y = starting_position['y']
    current_heading = starting_position['heading']
    current_time = imu_msgs[0][0].to_sec()

    velocity_x = 0.0
    velocity_y = 0.0

    imu_time_list = []
    imu_x_list = []
    imu_y_list = []
    imu_heading_list = []

    # Compute IMU-based trajectory
    for t, msg in imu_msgs:
        msg_sec = t.to_sec()
        dt = msg_sec - current_time
        if dt <= 0:
            continue

        angular_velocity_z = msg.angular_velocity.z
        linear_acceleration_x = msg.linear_acceleration.x
        linear_acceleration_y = msg.linear_acceleration.y

        current_heading += angular_velocity_z * dt

        accel_global_x = (linear_acceleration_x * math.cos(-current_heading) - linear_acceleration_y * math.sin(-current_heading))
        accel_global_y = (linear_acceleration_x * math.sin(-current_heading) + linear_acceleration_y * math.cos(-current_heading))

        velocity_x += accel_global_x * dt
        velocity_y += accel_global_y * dt

        current_x += velocity_x * dt
        current_y += velocity_y * dt
        current_time = msg_sec

        imu_time_list.append(msg_sec)
        imu_x_list.append(current_x)
        imu_y_list.append(current_y)
        imu_heading_list.append(current_heading)

    # Convert lists to numpy arrays
    imu_time_np = np.array(imu_time_list)
    imu_x_np = np.array(imu_x_list)
    imu_y_np = np.array(imu_y_list)
    imu_heading_np = np.array(imu_heading_list)

    # Create interpolation functions
    interp_x = interp1d(imu_time_np, imu_x_np, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(imu_time_np, imu_y_np, kind='linear', fill_value="extrapolate")
    interp_heading = interp1d(imu_time_np, imu_heading_np, kind='linear', fill_value="extrapolate")

    # Interpolate IMU data at odom timestamps
    for odom_time in odom_timestamps:
        odom_time_sec = odom_time.timestamp()  # Convert datetime to seconds
        x_values_IMU.append(interp_x(odom_time_sec))
        y_values_IMU.append(interp_y(odom_time_sec))
        heading_values_IMU.append(interp_heading(odom_time_sec))

    return x_values_IMU, y_values_IMU, heading_values_IMU



def plot_data(time_stamps, x_values, y_values, heading_values, filename):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Convert datetime to numerical format for matplotlib
    time_stamps_num = date2num(time_stamps)

    ax[0].plot(time_stamps_num, x_values[0], '-', label='x_gt')
    ax[0].plot(time_stamps_num, x_values[1], '-', label='x_pred_imu')
    ax[0].plot(time_stamps_num, x_values[2], '-', label='x_pred')
    ax[0].set_ylabel('X Value')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(time_stamps_num, y_values[0], '-', label='y_gt')
    ax[1].plot(time_stamps_num, y_values[1], '-', label='y_pred_imu')
    ax[1].plot(time_stamps_num, y_values[2], '-', label='y_pred')
    ax[1].set_ylabel('Y Value')
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(time_stamps_num, heading_values[0], '-', label='yaw_gt')
    ax[2].plot(time_stamps_num, heading_values[1], '-', label='yaw_pred_imu')
    ax[2].plot(time_stamps_num, heading_values[2], '-', label='yaw_pred')
    ax[2].set_ylabel('Heading (rad)')
    ax[2].set_xlabel('Time')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def prepare_data(odom_gt, imu_pred, seq_length):
    gt_set, pred_set = [], []
    x_pred, y_pred, yaw_pred = imu_pred
    x_gt_set, y_gt_set, yaw_gt_set = odom_gt
    num_samples = len(x_gt_set)

    for i in range(num_samples - seq_length + 1):  # Ensure full sequence extraction
        # Extract prediction window
        x = x_pred[i:i+seq_length]
        y = y_pred[i:i+seq_length]
        yaw = yaw_pred[i:i+seq_length]

        # Stack predictions into shape (seq_length, 3)
        pred_window = np.stack([x, y, yaw], axis=-1)  # Shape: (seq_length, 3)
        pred_set.append(pred_window)

        # Ground truth at index `i + seq_length - 1`
        x_gt, y_gt, yaw_gt = x_gt_set[i + seq_length - 1], y_gt_set[i + seq_length - 1], yaw_gt_set[i + seq_length - 1]
        gt_set.append([x_gt, y_gt, yaw_gt])  # Shape: (3,)

    return np.stack(pred_set), np.array(gt_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot x, y, and heading data from a ROS bag file.")
    parser.add_argument('bag', type=str, help="Path to the ROS bag file.")
    parser.add_argument('model', type=str, default=None, nargs='?', help="Path to the model file.") # optional
    args = parser.parse_args()

    bag_file_path = args.bag
    MODEL = args.model

    print("Extracting data from the bag file...")
    time_stamps, x_gt, y_gt, yaw_gt, imu_msgs = extract_data(bag_file_path)
    starting_position = {'x': x_gt[0], 'y': y_gt[0], 'heading': yaw_gt[0]}
    
    print("Computing position data based on IMU sensors...")
    x_pred_imu, y_prend_imu, yaw_pred_imu = compute_data(starting_position, imu_msgs, time_stamps)

    print("Creating the dataset...")
    seq_length = 50
    x, y = prepare_data(
        odom_gt=(x_gt, y_gt, yaw_gt), 
        imu_pred=(x_pred_imu, y_prend_imu, yaw_pred_imu),
        seq_length=seq_length
    )

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    # tensor must be in the form of (batch, seq, feature)

    dataX = Variable(torch.Tensor(np.array(x))).to('cuda')
    dataY = Variable(torch.Tensor(np.array(y))).to('cuda')

    trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).to('cuda')
    trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).to('cuda')

    testX = Variable(torch.Tensor(np.array(x[0:len(x)]))).to('cuda')
    testY = Variable(torch.Tensor(np.array(y[0:len(y)]))).to('cuda')

    # training

    num_epochs = 4000
    learning_rate = 0.008

    input_size = 3
    hidden_size = 20
    num_layers = 1

    output_size = 3

    lstm = LSTM(output_size, input_size, hidden_size, num_layers)
    lstm.to('cuda')

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    if not MODEL:
        print("training the model...")
        lstm.train_loop(num_epochs, trainX, trainY, optimizer, criterion)
        torch.save(lstm.state_dict(), 'model.pth')
    else:
        lstm.load_state_dict(torch.load(MODEL))

    # testing
    print("testing the model...")
    results = lstm.test_loop(testX, testY, train_size, criterion)

    x_pred, y_pred, yaw_pred = [], [], []
    for i in range(len(results)):
        x, y, yaw = [float(x) for x in results[i]]
        x_pred.append(x)
        y_pred.append(y)
        yaw_pred.append(yaw)
    
    # padd with last value at start
    x_pred = np.pad(x_pred, (seq_length - 1, 0), 'edge')
    y_pred = np.pad(y_pred, (seq_length - 1, 0), 'edge')
    yaw_pred = np.pad(yaw_pred, (seq_length - 1, 0), 'edge')

    plot_data(time_stamps, [x_gt, x_pred_imu, x_pred], [y_gt, y_prend_imu, y_pred], [yaw_gt, yaw_pred_imu, yaw_pred], 'output.png')
    exit()
    
    delta_list = []
    for epoch in range(num_epochs):
        x_pred, y_pred, yaw_pred = [], [], []
        for gt_odom, pred_window in dataset:

            gt_odom = gt_odom.to('cuda')
            pred_window = pred_window.to('cuda')


            outputs = lstm(pred_window)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, gt_odom)

            loss.backward()

            optimizer.step()

            x, y, yaw = [float(x) for x in outputs[0]]
            x_pred.append(x)
            y_pred.append(y)
            yaw_pred.append(yaw)

        #if epoch % 10 == 0:
            # x_gt, y_gt, yaw_gt = [float(x) for x in gt_odom_tensor[0]]
            # delta_x = abs(x - x_gt)
            # delta_y = abs(y - y_gt)
            # delta_yaw = abs(yaw - yaw_gt)
            # delta_list.append((delta_x, delta_y, delta_yaw))
            # print("Epoch: %d, loss: %1.5f, delta_x: %f, delta_y: %f, delta_yaw: %f" % (epoch, loss.item(), delta_x, delta_y, delta_yaw))
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        # remove the additional zeros
        x_pred = x_pred[:len(x_gt)]
        y_pred = y_pred[:len(y_gt)]
        yaw_pred = yaw_pred[:len(yaw_gt)]

        plot_data(time_stamps, [x_gt, x_pred_imu, x_pred], [y_gt, y_prend_imu, y_pred], [yaw_gt, yaw_pred_imu, yaw_pred], 'output.png')
    
    print("Training complete!")
