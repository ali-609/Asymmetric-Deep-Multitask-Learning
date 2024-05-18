import os
import glob
import json
import numpy as np

rel_dirs=sorted(glob.glob('./Datasets/gateway/*/bus/*bus_signals.json'))

dirs=absolute_paths = [os.path.abspath(path) for path in rel_dirs]

for file in dirs:
    print(file)


    os.chdir(os.path.dirname(file))


    json_file = open(file) 
    # json_data = json.load(json_file)
    with open(file) as json_file:
        json_data = json.load(json_file)


    for item in json_data:
        flexray_data = item.get('flexray', {})  # Safely retrieve 'flexray' key
        steering_data = flexray_data.get('steering_angle', {})  # Safely retrieve 'steering_angle_calculated' key
        steering_angle_sign = flexray_data.get('steering_angle_sign', {}) 

        steering_angles = np.array(steering_data.get('values',[]))  # Safely retrieve 'values' key
        steering_sign = np.array(steering_angle_sign.get('values', [])) 

        steering_angles[steering_sign == 0] *= -1
        
        output_file_path = item['frame_name']


        steering_angles = steering_angles.tolist()


        # Save the steering angles in JSON format to the file
        with open(output_file_path, 'w') as output_file:
            json.dump({"steering_angles": steering_angles}, output_file, indent=4)



# dirs
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180807_145028/bus/20180807145028_bus_signals.json'
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180810_142822/bus/20180810142822_bus_signals.json'
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180925_101535/bus/20180925101535_bus_signals.json'
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180925_112730/bus/20180925112730_bus_signals.json'
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180925_124435/bus/20180925124435_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20180925_135056/bus/20180925135056_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181008_095521/bus/20181008095521_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181016_082154/bus/20181016082154_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181016_125231/bus/20181016125231_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181107_132300/bus/20181107132300_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181107_132730/bus/20181107132730_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181107_133258/bus/20181107133258_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181107_133445/bus/20181107133445_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181108_084007/bus/20181108084007_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181108_091945/bus/20181108091945_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181108_103155/bus/20181108103155_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181108_123750/bus/20181108123750_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181108_141609/bus/20181108141609_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181204_135952/bus/20181204135952_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181204_154421/bus/20181204154421_bus_signals.json' 
#  '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/20181204_170238/bus/20181204170238_bus_signals.json'