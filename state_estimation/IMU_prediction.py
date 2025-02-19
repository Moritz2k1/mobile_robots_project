import rosbag
import math
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num
from datetime import datetime
from model import LSTM
import argparse
from scipy.interpolate import interp1d
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

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
    angular_velocity_z_interp = []
    linear_acceleration_x_interp = []
    linear_acceleration_y_interp = []
    for odom_time in odom_timestamps:
        odom_time_sec = odom_time.timestamp()  # Convert datetime to seconds
        x_values_IMU.append(interp_x(odom_time_sec))
        y_values_IMU.append(interp_y(odom_time_sec))
        heading_values_IMU.append(interp_heading(odom_time_sec))

    return x_values_IMU, y_values_IMU, heading_values_IMU

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_data(time_stamps, train_size, x_values, y_values, heading_values, filename):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Convert datetime to numerical format for matplotlib
    time_stamps_num = date2num(time_stamps)
    time_stamp_split = time_stamps_num[train_size]

    ax[0].plot(time_stamps_num, x_values[0], '-', label='x_gt')
    ax[0].plot(time_stamps_num, x_values[1], '-', label='x_pred_imu')
    ax[0].plot(time_stamps_num, x_values[2], '-', label='x_pred')
    ax[0].plot(time_stamps_num, smooth(x_values[2], 5), '-', label='x_pred_smooth')
    ax[0].axvline(x=time_stamp_split, c='r', linestyle='--')
    ax[0].set_ylabel('X Value')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(time_stamps_num, y_values[0], '-', label='y_gt')
    ax[1].plot(time_stamps_num, y_values[1], '-', label='y_pred_imu')
    ax[1].plot(time_stamps_num, y_values[2], '-', label='y_pred')
    ax[1].plot(time_stamps_num, smooth(y_values[2], 5), '-', label='y_pred_smooth')
    ax[1].axvline(x=time_stamp_split, c='r', linestyle='--')
    ax[1].set_ylabel('Y Value')
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(time_stamps_num, heading_values[0], '-', label='yaw_gt')
    ax[2].plot(time_stamps_num, heading_values[1], '-', label='yaw_pred_imu')
    ax[2].plot(time_stamps_num, heading_values[2], '-', label='yaw_pred')
    ax[2].plot(time_stamps_num, smooth(heading_values[2], 5), '-', label='yaw_pred_smooth')
    ax[2].axvline(x=time_stamp_split, c='r', linestyle='--')
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
        pred_window = np.stack([x, y, yaw], axis=-1)  # Shape: (seq_length, 6)
        pred_set.append(pred_window)

        # Ground truth at index `i + seq_length - 1`
        x_gt, y_gt, yaw_gt = x_gt_set[i + seq_length - 1], y_gt_set[i + seq_length - 1], yaw_gt_set[i + seq_length - 1]
        gt_set.append([x_gt, y_gt, yaw_gt])  # Shape: (3,)

    return np.stack(pred_set), np.array(gt_set)

def train(model, num_epochs, all_x, all_y, train_split, optimizer, criterion, lr_scheduler):
    delta_list = []
    best_loss = float('inf')
    best_state_dict = None
    for epoch in range(num_epochs):
        total_loss = 0
        for dataX, dataY in zip(all_x, all_y):
            train_size = int(len(dataY) * train_split)
            trainX, trainY = dataX[:train_size, :, :], dataY[:train_size, :]
            outputs = model(trainX)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, trainY)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
        if total_loss < best_loss:
            best_loss = total_loss
            best_state_dict = model.state_dict()
        lr_scheduler.step()
        if epoch % 100 == 0:
            # delta = sum(abs(outputs - trainY)).item()
            # delta_list.append(delta)
            # print("Epoch: %d, loss: %1.5f, delta: %f" % (epoch, loss.item(), delta))
            print("Epoch: %d, last_loss: %1.5f, lr: %f" % (epoch, total_loss, optimizer.param_groups[0]['lr']))
    print("Best loss: %1.5f" % (best_loss))
    model.load_state_dict(best_state_dict)
    # print("Median delta: %f" % (np.median(delta_list)))

def test(model, dataX, dataY, criterion):
    # set model to evaluation mode
    model.eval()

    train_predict = model(dataX)
    mserror = criterion(train_predict, dataY)
    print("MSE for val data: %1.5f" % (mserror.item()))
    #print("Delta for val data: %f" % (sum(abs(train_predict - dataY))))
    return train_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot x, y, and heading data from a ROS bag file.")
    parser.add_argument('bag_folder', type=str, help="Path to the ROS bag folder.")
    parser.add_argument('model', type=str, default=None, nargs='?', help="Path to the model file.") # optional
    args = parser.parse_args()

    BAG_FOLDER_PATH = args.bag_folder
    MODEL = args.model

    seq_length = 10
    all_time_stamps = []
    all_gt_x, all_gt_y, all_gt_yaw = [], [], []
    all_imu_x, all_imu_y, all_imu_yaw = [], [], []
    all_x, all_y = [], []

    list_scaler_x, list_scaler_y, list_scaler_yaw = [], [], []

    print("Extracting data from the bag file...")
    for bag_file_path in os.listdir(BAG_FOLDER_PATH):
        if not bag_file_path.endswith('.bag'):
            continue
        print(f"Processing {bag_file_path}...")
        time_stamps, x_gt, y_gt, yaw_gt, imu_msgs = extract_data(os.path.join(BAG_FOLDER_PATH, bag_file_path))
        all_time_stamps.append(time_stamps)
        all_gt_x.append(x_gt)
        all_gt_y.append(y_gt)
        all_gt_yaw.append(yaw_gt)
        starting_position = {'x': x_gt[0], 'y': y_gt[0], 'heading': yaw_gt[0]}
        x_pred_imu, y_pred_imu, yaw_pred_imu = compute_data(starting_position, imu_msgs, time_stamps)

        all_imu_x.append(x_pred_imu)
        all_imu_y.append(y_pred_imu)
        all_imu_yaw.append(yaw_pred_imu)

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_yaw = MinMaxScaler()

        x_gt = scaler_x.fit_transform(np.array(x_gt).reshape(-1, 1)).flatten()
        y_gt = scaler_y.fit_transform(np.array(y_gt).reshape(-1, 1)).flatten()
        yaw_gt = scaler_yaw.fit_transform(np.array(yaw_gt).reshape(-1, 1)).flatten()

        x_pred_imu = scaler_x.transform(np.array(x_pred_imu).reshape(-1, 1)).flatten()
        y_pred_imu = scaler_y.transform(np.array(y_pred_imu).reshape(-1, 1)).flatten()
        yaw_pred_imu = scaler_yaw.transform(np.array(yaw_pred_imu).reshape(-1, 1)).flatten()

        list_scaler_x.append(scaler_x)
        list_scaler_y.append(scaler_y)
        list_scaler_yaw.append(scaler_yaw)

        x, y = prepare_data(
            odom_gt=(x_gt, y_gt, yaw_gt),
            imu_pred=(x_pred_imu, y_pred_imu, yaw_pred_imu),
            seq_length=seq_length
        )

        dataX = Variable(torch.Tensor(np.array(x))).to('cuda')
        dataY = Variable(torch.Tensor(np.array(y))).to('cuda')

        all_x.append(dataX)
        all_y.append(dataY)

    # Combine datasets from all bags
    

    train_split = 0.8

    # training
    num_epochs = 30000
    learning_rate = 0.007

    hidden_size = 10
    num_layers = 3

    lstm = LSTM(hidden_size, num_layers)
    lstm.to('cuda')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # linear lr
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)

    if not MODEL:
        print("training the model...")
        train(lstm, num_epochs, all_x, all_y, train_split, optimizer, criterion, lr_scheduler)
        torch.save(lstm.state_dict(), 'model.pth')
        print("Training complete!")
    else:
        lstm.load_state_dict(torch.load(MODEL))

    # testing
    print("testing the model...")
    all_results = []
    for dataX, dataY in zip(all_x, all_y):
        results = test(lstm, dataX, dataY, criterion)
        all_results.append(results)

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

    for i, (x_gt, y_gt, yaw_gt, x_pred_imu, y_prend_imu, yaw_pred_imu, time_stamps, results, scaler_x, scaler_y, scaler_yaw) in enumerate(zip(all_gt_x, all_gt_y, all_gt_yaw, all_imu_x, all_imu_y, all_imu_yaw, all_time_stamps, all_results, list_scaler_x, list_scaler_y, list_scaler_yaw)):
        # Number of results corresponding to this bag
        num_samples = len(x_gt) - seq_length + 1

        # Extract model predictions for this bag
        x_pred, y_pred, yaw_pred = [], [], []
        for j in range(len(results)):
            x, y, yaw = [float(val) for val in results[j]]
            x_pred.append(x)
            y_pred.append(y)
            yaw_pred.append(yaw)

        # Pad predictions to align with original timestamps
        x_pred = np.pad(x_pred, (seq_length - 1, 0), 'edge')
        y_pred = np.pad(y_pred, (seq_length - 1, 0), 'edge')
        yaw_pred = np.pad(yaw_pred, (seq_length - 1, 0), 'edge')
        train_size = int(len(time_stamps) * train_split)
        
        x_pred = scaler_x.inverse_transform(np.array(x_pred).reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
        yaw_pred = scaler_yaw.inverse_transform(np.array(yaw_pred).reshape(-1, 1)).flatten()

        plot_data(time_stamps, train_size, [x_gt, x_pred_imu, x_pred], [y_gt, y_prend_imu, y_pred], [yaw_gt, yaw_pred_imu, yaw_pred], 'output_' + str(i) + '.png')
