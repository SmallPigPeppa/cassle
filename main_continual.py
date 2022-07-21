import copy
import itertools
import subprocess
import sys
import os
import json


def str_to_dict(command):
    d = {}
    for part, part_next in itertools.zip_longest(command[:-1], command[1:]):
        if part[:2] == "--":
            if part_next[:2] != "--":
                d[part] = part_next
            else:
                d[part] = part
        elif part[:2] != "--" and part_next[:2] != "--":
            part_prev = list(d.keys())[-1]
            if not isinstance(d[part_prev], list):
                d[part_prev] = [d[part_prev]]
            if not part_next[:2] == "--":
                d[part_prev].append(part_next)
    return d


def dict_to_list(command):
    s = []
    for k, v in command.items():
        s.append(k)
        if k != v and v[:2] != "--":
            s.append(v)
    return s


def run_bash_command(args):
    for i, a in enumerate(args):
        if isinstance(a, list):
            args[i] = " ".join(a)
    command = ("python3 main_pretrain.py", *args)
    command = " ".join(command)
    p = subprocess.Popen(command, shell=True)
    p.wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    args = str_to_dict(args)

    # parse args from the script
    num_tasks = int(args["--num_tasks"])
    start_task_idx = int(args.get("--task_idx", 0))
    distill_args = {k: v for k, v in args.items() if "distill" in k}

    # delete things that shouldn't be used for task_idx 0
    args.pop("--task_idx", None)
    for k in distill_args.keys():
        args.pop(k, None)

    # check if this experiment is being resumed
    # look for the file last_checkpoint.txt
    last_checkpoint_file = os.path.join(args["--checkpoint_dir"], "last_checkpoint.txt")
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file) as f:
            ckpt_path, args_path = [line.rstrip() for line in f.readlines()]
            start_task_idx = json.load(open(args_path))["task_idx"]
            args["--resume_from_checkpoint"] = ckpt_path

    # # main task loop
    # for task_idx in range(start_task_idx, num_tasks):
    #     print(f"\n#### Starting Task {task_idx} ####")
    #
    #     task_args = copy.deepcopy(args)
    #
    #     # add pretrained model arg
    #     if task_idx != 0 and task_idx != start_task_idx:
    #         task_args.pop("--resume_from_checkpoint", None)
    #         task_args.pop("--pretrained_model", None)
    #         assert os.path.exists(last_checkpoint_file)
    #         ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
    #         task_args["--pretrained_model"] = ckpt_path
    #
    #     if task_idx != 0 and distill_args:
    #         task_args.update(distill_args)
    #
    #     # add use_expansion and re_reparameterize
    #
    #     if task_idx in [1,3,4]:
    #         task_args['--use_expansion'] = '   '
    #     # use re_paramaterize after task1
    #     if task_idx in [2]:
    #         task_args['--re_paramaterize'] = '   '
    #
    #     # if task_idx == 1 :
    #     #     task_args['--use_expansion'] = '   '
    #     # # use re_paramaterize after task1
    #     # if task_idx ==2:
    #     #     task_args['--re_paramaterize'] = '   '
    #
    #     task_args["--task_idx"] = str(task_idx)
    #     task_args = dict_to_list(task_args)
    #
    #     run_bash_command(task_args)

    # main task loop
    ckpt_set=[]
    for task_idx in range(start_task_idx, num_tasks):
        print(f"\n#### Starting Task {task_idx} ####")

        if task_idx == 0:
            task_args = copy.deepcopy(args)
            task_args["--task_idx"] = str(task_idx)
            task_args = dict_to_list(task_args)
            run_bash_command(task_args)
        elif task_idx ==1:
            # use expansion
            task_args = copy.deepcopy(args)
            if task_idx != 0 and task_idx != start_task_idx:
                task_args.pop("--resume_from_checkpoint", None)
                task_args.pop("--pretrained_model", None)
                assert os.path.exists(last_checkpoint_file)
                ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
                task_args["--pretrained_model"] = ckpt_path
            # task_args.pop("--resume_from_checkpoint", None)
            # task_args.pop("--pretrained_model", None)
            # task_args.pop("--fixed_pretrained_model", None)
            # assert os.path.exists(last_checkpoint_file)
            # ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
            # task_args["--pretrained_model"] = ckpt_path

            task_args['--use_expansion'] = '   '
            # use re_paramaterize after task1
            if task_idx != 0 and distill_args:
                task_args.update(distill_args)
            task_args["--task_idx"] = str(task_idx)
            task_args = dict_to_list(task_args)
            run_bash_command(task_args)

            # use distill
            task_args = copy.deepcopy(args)
            task_args.pop("--resume_from_checkpoint", None)
            task_args.pop("--pretrained_model", None)
            task_args.pop("--fixed_pretrained_model", None)
            assert os.path.exists(last_checkpoint_file)
            # use task_n-1 as ckpt
            fixed_ckpt_path = ckpt_path
            ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
            task_args["--fixed_pretrained_model"] = fixed_ckpt_path
            task_args["--pretrained_model"] = ckpt_path
            task_args["--distiller"] = 'contrastive'
            task_args["--task_idx"] = str(task_idx)
            task_args['--re_paramaterize'] = '   '
            task_args = dict_to_list(task_args)
            run_bash_command(task_args)
        else:
            task_args = copy.deepcopy(args)
            if task_idx != 0 and task_idx != start_task_idx:
                task_args.pop("--resume_from_checkpoint", None)
                task_args.pop("--pretrained_model", None)
                task_args.pop("--fixed_pretrained_model", None)
                assert os.path.exists(last_checkpoint_file)
                ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
                task_args["--pretrained_model"] = ckpt_path

            # ckpt_set.append(ckpt_path)
            # task_args["--fixed_pretrained_model"] = ckpt_set[-1]
            # use task_n-1 as ckpt
            task_args["--fixed_pretrained_model"] = '/home/admin/code/cassle_v32.0/experiments/2022_07_20_21_07_33-simclr-cifar100/2l9dhj6d/simclr-cifar100-task2-ep=499-2l9dhj6d.ckpt'
            task_args["--distiller"] = 'contrastive'
            task_args["--task_idx"] = str(task_idx)
            task_args['--re_paramaterize'] = '   '
            task_args = dict_to_list(task_args)
            run_bash_command(task_args)

