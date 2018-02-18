# _*_ coding:utf-8 _*_

import sys
import inspect
import os
import shutil
import itertools as it
from extract_scene import is_scene, get_module
from constants import ANIMATIONS_DIR, STAGED_SCENES_DIR

# 对从模块中得到的子场景进行排序
def get_sorted_scene_names(module_name):
    module = get_module(module_name)
    line_to_scene = {}
    for name, scene_class in inspect.getmembers(module, is_scene):
        lines, line_no = inspect.getsourcelines(scene_class)
        line_to_scene[line_no] = name
    return [
        line_to_scene[line_no]
        for line_no in sorted(line_to_scene.keys())
    ]

# staged分段的动画
def stage_animations(module_name):
    scene_names = get_sorted_scene_names(module_name)
    animation_dir = os.path.join(
        ANIMATIONS_DIR, module_name.replace(".py", "")
    )
    files = os.listdir(animation_dir)
    sorted_files = []
    # 拼接存放每个分段动画路径
    for scene in scene_names:
        for clip in filter(lambda f : f.startswith(scene), files):
            sorted_files.append(
                os.path.join(animation_dir, clip)
            )
    staged_scenes_dir = os.path.join(animation_dir, "staged_scenes")
    count = 0
    # 建立一个新的分段动画目录
    while True:
        staged_scenes_dir = os.path.join(
            animation_dir, "staged_scenes_%d"%count
        )
        if not os.path.exists(staged_scenes_dir):
            os.makedirs(staged_scenes_dir)
            break
        #Otherwise, keep trying new names until 
        #there is a free one
        count += 1
    # 将分段的动画在这个分段动画目录进行组合
    for count, f in enumerate(sorted_files):
        symlink_name = os.path.join(
            staged_scenes_dir,
            "Scene_%03d"%count + f.split(os.sep)[-1]
        )
        # 创建一个软链接 symlink(src, dst) 分别是源地址和目标地址
        os.symlink(f, symlink_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("No module given.")
    module_name = sys.argv[1]
    stage_animations(module_name)