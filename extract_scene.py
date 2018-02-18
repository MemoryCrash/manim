#!/usr/bin/env python2
# _*_ coding:utf-8 _*_

import sys
# import getopt
import argparse
import imp
import itertools as it
import inspect
import traceback
import imp
import os
import subprocess as sp

from helpers import *
from scene import Scene
from camera import Camera

HELP_MESSAGE = """
   Usage:
   python extract_scene.py <module> [<scene name>]

   -p preview in low quality
   -s show and save picture of last frame
   -w write result to file [this is default if nothing else is stated]
   -l use low quality
   -m use medium quality
   -a run and save every scene in the script, or all args for the given scene
   -q don't print progress
   -f when writing to a movie file, export the frames in png sequence
   -t use transperency when exporting images
"""
SCENE_NOT_FOUND_MESSAGE = """
   That scene is not in the script
"""
CHOOSE_NUMBER_MESSAGE = """
Choose number corresponding to desired scene/arguments.
(Use comma separated list for multiple entries)

Choice(s): """
INVALID_NUMBER_MESSAGE = "Fine then, if you don't want to give a valid number I'll just quit"

NO_SCENE_MESSAGE = """
   There are no scenes inside that module
"""


def get_configuration():
   try:
       # argparse是python自带的命令行解析库
      parser = argparse.ArgumentParser()
       # add_argument()用来指定需要从命令行接收的参数
      parser.add_argument(
         "file", help = "path to file holding the python code for the scene"
      )
      parser.add_argument(
         "scene_name", help = "Name of the Scene class you want to see"
      )
       # 命令行分为短参数和长参数，下面的代码是指定了短参数和长参数的映射关系
      optional_args = [
         ("-p", "--preview"),
         ("-w", "--write_to_movie"),
         ("-s", "--show_last_frame"),
         ("-l", "--low_quality"),
         ("-m", "--medium_quality"),
         ("-g", "--save_pngs"),
         ("-f", "--show_file_in_finder"),
         ("-t", "--transparent"),
         ("-q", "--quiet"),
         ("-a", "--write_all")
      ]
       # 将这些参数添加到接收目标中去
      for short_arg, long_arg in optional_args:
          # 这里action="store_true"的含义是如果某个参数被指定了则将true赋值给
         parser.add_argument(short_arg, long_arg, action = "store_true")
      # 这里没有加action的参数依然接收具体的参数
      parser.add_argument("-o", "--output_name")
      parser.add_argument("-n", "--skip_to_animation_number")
       # parser.parse_args()进行参数解析
      args = parser.parse_args()
   except argparse.ArgumentError as err:
      print(str(err))
      sys.exit(2)
       # 将配置从命令行获取到的参数信息赋值到config中保存。
   config = {
      "file"            : args.file,
      #"file": 'example_scenes.py',
      "scene_name"      : args.scene_name,
      #"scene_name": 'SquareToCircle',
      "open_video_upon_completion" : args.preview,
      "show_file_in_finder" : args.show_file_in_finder,
       # 默认写到movie中去
      #By default, write to file
      "write_to_movie"  : args.write_to_movie or not args.show_last_frame,
      "show_last_frame" : args.show_last_frame,
      "save_pngs"       : args.save_pngs,
      #If -t is passed in (for transparent), this will be RGBA
      "saved_image_mode": "RGBA" if args.transparent else "RGB",
      "quiet"           : args.quiet or args.write_all,
      "ignore_waits"    : args.preview,
      #"ignore_waits": True,
      "write_all"       : args.write_all,
      "output_name"     : args.output_name,
      "skip_to_animation_number" : args.skip_to_animation_number,
   }
   # 对一些参数判断触发状态后进行单独的设置
   if args.low_quality:
       # 低质量设置相片大小为(480, 854)
      config["camera_config"] = LOW_QUALITY_CAMERA_CONFIG
       # 低质量设置1秒15帧
      config["frame_duration"] = LOW_QUALITY_FRAME_DURATION
   elif args.medium_quality:
       # 中质量设置相片大小为(720, 1280)
      config["camera_config"] = MEDIUM_QUALITY_CAMERA_CONFIG
       # 中质量设置1秒30帧
      config["frame_duration"] = MEDIUM_QUALITY_FRAME_DURATION
   else:
       # 高质量设置相片大小为(1080, 1920)
      config["camera_config"] = PRODUCTION_QUALITY_CAMERA_CONFIG
       # 高质量设置1秒60帧
      config["frame_duration"] = PRODUCTION_QUALITY_FRAME_DURATION
    # 对希望跳过的动画编号进行字符串到数值的转换
   stan = config["skip_to_animation_number"]
   if stan is not None:
      config["skip_to_animation_number"] = int(stan)
    #any([])若数组中每个数据都不都为false，0， none则返回true
    #目的就是判断是否存在跳过动画的情况
   config["skip_animations"] = any([
      config["show_last_frame"] and not config["write_to_movie"],
      config["skip_to_animation_number"],
   ])
   return config

def handle_scene(scene, **config):
   if config["quiet"]:
      curr_stdout = sys.stdout
      sys.stdout = open(os.devnull, "w")

   if config["show_last_frame"]:
      scene.save_image(mode = config["saved_image_mode"])
    # 以下三个是打开文件的条件
   open_file = any([
      config["show_last_frame"],
      config["open_video_upon_completion"],
      config["show_file_in_finder"]
   ])
   if open_file:
      commands = ["open"]
      if config["show_file_in_finder"]:
         commands.append("-R")
      #
      if config["show_last_frame"]:
         commands.append(scene.get_image_file_path())
      else:
         commands.append(scene.get_movie_file_path())
      sp.call(commands)

   if config["quiet"]:
      sys.stdout.close()
      sys.stdout = curr_stdout

# 目的是对scene的子类返回true
def is_scene(obj):
   if not inspect.isclass(obj):
      return False
   if not issubclass(obj, Scene):
      return False
   if obj == Scene:
      return False
   return True

# 打印子场景选项
def prompt_user_for_choice(name_to_obj):
   num_to_name = {}
   names = sorted(name_to_obj.keys())
   for count, name in zip(it.count(1), names):
      print("%d: %s"%(count, name))
      num_to_name[count] = name
   try:
      user_input = raw_input(CHOOSE_NUMBER_MESSAGE)
      return [
         name_to_obj[num_to_name[int(num_str)]]
         for num_str in user_input.split(",")
      ]
   except:
      print(INVALID_NUMBER_MESSAGE)
      sys.exit()

# 获取场景子类
def get_scene_classes(scene_names_to_classes, config):
   if len(scene_names_to_classes) == 0:
      print(NO_SCENE_MESSAGE)
      return []
   if len(scene_names_to_classes) == 1:
      return scene_names_to_classes.values()
   #如果存在多个场景则返回命令行中提到的场景
   if config["scene_name"] in scene_names_to_classes:
      return [scene_names_to_classes[config["scene_name"]] ]
   if config["scene_name"] != "":
      print(SCENE_NOT_FOUND_MESSAGE)
      return []
   # 如果是write_all则返回所有的子场景
   if config["write_all"]:
      return scene_names_to_classes.values()
   # 最后的场景是将所有的场景在屏幕打印出来让用户选择他希望运行的子场景
   return prompt_user_for_choice(scene_names_to_classes)

# imp 模块提供了一个import功能
# 从windows系统获取模块名
def get_module_windows(file_name):
   module_name = file_name.replace(".py", "")
   # 在当前路径寻找__init__模块并进行加载，这里使用*的作用是将list对应填充到参数
   last_module = imp.load_module("__init__", *imp.find_module("__init__", ['.']))
   # 对模块名称进行分割，可能有多个模块
   for part in module_name.split(os.sep):
      load_args = imp.find_module(part, [os.path.dirname(last_module.__file__)])
      last_module = imp.load_module(part, *load_args)
   return last_module

# 从linux系统获取模块名
def get_module_posix(file_name):
    module_name = file_name.replace(".py", "")
    last_module = imp.load_module(".", *imp.find_module("."))
    for part in module_name.split(os.sep):
        load_args = imp.find_module(part, last_module.__path__)
        last_module = imp.load_module(part, *load_args)
    return last_module

#获取模块名
def get_module(file_name):
    # 判断所处的操作系统，windows系统返回nt，linux系统返回posix
    if os.name == 'nt':
        return get_module_windows(file_name)
    return get_module_posix(file_name)

# 主函数的入口，获取命令行中的参数信息
def main():
    # 获取外部配置信息
   config = get_configuration()

    # 获取命令行参数中指定的模块名，也就是文件
   module = get_module(config["file"])

    # 将模块中的scene子类信息提取出来 is_scene用来进行过滤
   scene_names_to_classes = dict(
      inspect.getmembers(module, is_scene)
   )

    # 构造输入视频的路径
   config["output_directory"] = os.path.join(
      ANIMATIONS_DIR,
      config["file"].replace(".py", "")
   )

    # 组合屏幕需要的参数配置
   scene_kwargs = dict([
      (key, config[key])
      for key in [
         "camera_config",
         "frame_duration",
         "skip_animations",
         "write_to_movie",
         "output_directory",
         "save_pngs",
         "skip_to_animation_number",
      ]
   ])
   
   scene_kwargs["name"] = config["output_name"]
   if config["save_pngs"]:
      print "We are going to save a PNG sequence as well..."
      scene_kwargs["save_pngs"] = True
      scene_kwargs["pngs_mode"] = config["saved_image_mode"]

    #遍历所有的子场景
   for SceneClass in get_scene_classes(scene_names_to_classes, config):
      try:
          # 执行子场景，子场景在构造函数中便被执行
         handle_scene(SceneClass(**scene_kwargs), **config)
         # 播放正常结束音
         play_finish_sound()
      except:
         print("\n\n")
         traceback.print_exc()
         print("\n\n")
         # 播放失败音乐
         play_error_sound()


if __name__ == "__main__":
   main()
