"""
example usage:
python convert_tflite.py \
    name_of_model \
    -ml

-m converts normal model
-l converts tflite and edgetpu model

python convert_tflite.py my_ssd_mobilenet_v1_300x300_raspictures -ml
"""
import subprocess
import os
import re
import argparse
from shutil import copyfile


parser = argparse.ArgumentParser(description='Which model should be converted?')
parser.add_argument('Model', metavar='model', type=str,
                    help='the name of the model folder, should be in current cwd')
parser.add_argument('-m',
                    '--model',
                    action='store_true',
                    help='Freeze the normal model')
parser.add_argument('-l',
                    '--lite',
                    action='store_true',
                    help='Freeze the tflite model and convert to edgetpu')
args = parser.parse_args()

model = args.Model
modelpath = os.path.join("mymodels",model)
exportpath = os.path.join("exported", "export_"+model)
exporttflitepath = os.path.join("tflite_exported","export_tflite_"+model)

pattern = r'model.ckpt-(\d*).meta'
ckpts = [re.search(pattern, x).group(1) for x in os.listdir(modelpath) if re.search(pattern, x)]
max_ckpt = max(ckpts)
print(f"highest checkpoint found: {max_ckpt}")
cwd = os.getcwd()
os.environ["PYTHONPATH"] = f"{cwd}/models/research/slim"


if args.model:
    print('freezing normal graph')
    process_convert = subprocess.run(
        ['python',
        'export_inference_graph.py',
        '--input_type', 'image_tensor',
        '--pipeline_config_path', modelpath+'/pipeline.config',
        '--trained_checkpoint_prefix', f'{modelpath}/model.ckpt-{max_ckpt}',
        '--output_directory', exportpath])

if args.lite:
    print('freezing tflite graph')
    process_freeze_tflite = subprocess.run(
        [
            'python', 'export_tflite_ssd_graph.py',
            '--pipeline_config_path', modelpath+'/pipeline.config',
            '--trained_checkpoint_prefix', f'{modelpath}/model.ckpt-{max_ckpt}',
            '--output_directory', exporttflitepath
        ]
    )

    print('converting tflite model')
    process_convert_tflite = subprocess.run(
        [   
            'tflite_convert', 
            '--graph_def_file', f'{exporttflitepath}/tflite_graph.pb',
            f'--output_file=./{exporttflitepath}/detect.tflite',
            '--output_format=TFLITE',
            '--input_shapes=1,300,300,3',
            '--input_arrays=normalized_input_image_tensor',
            "--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3",
            '--inference_type=QUANTIZED_UINT8',
            '--mean_values=128',
            '--std_dev_values=127',
            '--change_concat_input_ranges=false', 
            '--allow_custom_ops'
        ]
    )

    print('compiling edgetpu model')
    
    process_compile_edgetpu = subprocess.run(
        [
            'docker', 'run', '--rm' , 
            '-v', f'{cwd}/{exporttflitepath}/:/data/',
            'edgetpu_compiler' ,'edgetpu_compiler', 'detect.tflite'
        ]
    )

    labels = []
    with open('/Users/matt/offhaw/quantized_tf1_switch_example/mymodels/training/car-bicycle_label_map.pbtxt') as f:
        s=f.readlines()
    for l in s:
        if 'name' in l:
            print(l)
            labels.append(re.search("'(.*?)'",l).group(1))
    with open(f'{cwd}/{exporttflitepath}/labelmap.txt', 'w') as f:
        f.write('\n'.join(labels))




