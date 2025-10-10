import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval/view_synthesis")
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--solver_type', type=str, default='hybrid')
args = parser.parse_args()

all_scenes = sorted(os.listdir(args.dataset))

if not args.skip_training:
    common_args = f" --quiet --eval --solver_type {args.solver_type}"
    
    for scene in all_scenes:
        source = os.path.join(args.dataset, scene)
        model = os.path.join(args.output_path, scene)
        os.system("python train.py -s " + source + " -m " + model + common_args + (" --lambda_dist 1000" if not 'cornell' in scene else ""))

if not args.skip_rendering:
    common_args = f" --solver_type {args.solver_type}"
    for scene in all_scenes:
        source = os.path.join(args.dataset, scene)
        model = os.path.join(args.output_path, scene)
        os.system("python render.py" + " -m " + model + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        model = os.path.join(args.output_path, scene)
        command = "python metrics.py -m " + model
        if not 'Cup' in scene and not 'Egg' in scene and not 'Fabric' in scene and not 'Tower' in scene:
            command += " --gamma"
        os.system(command)