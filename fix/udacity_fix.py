import rosbag

bag = rosbag.Bag('/gpfs/space/home/alimahar/hydra/Datasets/Udacity/dataset.bag')
# Topics:
# /can_bus_dbw/can_rx
# /center_camera/image_color
# /diagnostics
# /left_camera/image_color
# /right_camera/image_color
# /tf
# /vehicle/brake_info_report
# /vehicle/brake_report
# /vehicle/dbw_enabled
# /vehicle/filtered_accel
# /vehicle/fuel_level_report
# /vehicle/gear_report
# /vehicle/gps/fix
# /vehicle/gps/time
# /vehicle/gps/vel
# /vehicle/imu/data_raw
# /vehicle/joint_states
# /vehicle/misc_1_report
# /vehicle/sonar_cloud
# /vehicle/steering_report
# /vehicle/surround_report
# /vehicle/suspension_report
# /vehicle/throttle_info_report
# /vehicle/throttle_report
# /vehicle/tire_pressure_report
# /vehicle/twist_controller/parameter_descriptions
# /vehicle/twist_controller/parameter_updates
# /vehicle/wheel_speed_report




for topic, msg, t in bag.read_messages():
    print(f"Topic: {topic}")
    print(f"Message type: {msg._type}")
    print("-----------------------")

bag.close()
