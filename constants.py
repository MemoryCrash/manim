# _*_ coding:utf-8 _*_

import os
import numpy as np

DEFAULT_HEIGHT = 1080
DEFAULT_WIDTH  = 1920

LOW_QUALITY_FRAME_DURATION = 1./15
MEDIUM_QUALITY_FRAME_DURATION = 1./30
PRODUCTION_QUALITY_FRAME_DURATION = 1./60

#There might be other configuration than pixel_shape later...
PRODUCTION_QUALITY_CAMERA_CONFIG = {
    "pixel_shape" : (DEFAULT_HEIGHT, DEFAULT_WIDTH),
}

MEDIUM_QUALITY_CAMERA_CONFIG = {
    "pixel_shape" : (720, 1280),
}

LOW_QUALITY_CAMERA_CONFIG = {
    "pixel_shape" : (480, 854),
}

DEFAULT_POINT_DENSITY_2D = 25 
DEFAULT_POINT_DENSITY_1D = 250

DEFAULT_POINT_THICKNESS = 4

#TODO, Make sure these are not needed
SPACE_HEIGHT = 4.0
SPACE_WIDTH = SPACE_HEIGHT * DEFAULT_WIDTH / DEFAULT_HEIGHT


SMALL_BUFF = 0.1
MED_SMALL_BUFF = 0.25
MED_LARGE_BUFF = 0.5
LARGE_BUFF = 1

DEFAULT_MOBJECT_TO_EDGE_BUFFER = MED_LARGE_BUFF
DEFAULT_MOBJECT_TO_MOBJECT_BUFFER = MED_SMALL_BUFF


#All in seconds
DEFAULT_ANIMATION_RUN_TIME = 1.0
DEFAULT_POINTWISE_FUNCTION_RUN_TIME = 3.0
DEFAULT_WAIT_TIME = 1.0


ORIGIN = np.array(( 0., 0., 0.))
UP     = np.array(( 0., 1., 0.))
DOWN   = np.array(( 0.,-1., 0.))
RIGHT  = np.array(( 1., 0., 0.))
LEFT   = np.array((-1., 0., 0.))
IN     = np.array(( 0., 0.,-1.))
OUT    = np.array(( 0., 0., 1.))

TOP        = SPACE_HEIGHT*UP
BOTTOM     = SPACE_HEIGHT*DOWN
LEFT_SIDE  = SPACE_WIDTH*LEFT
RIGHT_SIDE = SPACE_WIDTH*RIGHT

TAU = 2*np.pi

# Change this to point to where you want 
# animation files to output
MEDIA_DIR = os.path.join("/Users/dailei/Documents/ori_manim/manim/", "media/")

ANIMATIONS_DIR = os.path.join(MEDIA_DIR, "animations")
RASTER_IMAGE_DIR = os.path.join(MEDIA_DIR, "designs", "raster_images")
SVG_IMAGE_DIR = os.path.join(MEDIA_DIR, "designs", "svg_images")
#TODO, staged scenes should really go into a subdirectory of a given scenes directory
STAGED_SCENES_DIR = os.path.join(ANIMATIONS_DIR, "staged_scenes") 
###
THIS_DIR          = os.path.dirname(os.path.realpath(__file__))
FILE_DIR          = os.path.join(THIS_DIR, "files")
TEX_DIR           = os.path.join(FILE_DIR, "Tex")
TEX_IMAGE_DIR     = TEX_DIR #TODO, What is this doing?
#These two may be depricated now.
MOBJECT_DIR       = os.path.join(FILE_DIR, "mobjects")
IMAGE_MOBJECT_DIR = os.path.join(MOBJECT_DIR, "image")

if not os.path.exists(MEDIA_DIR):
    raise Exception("""
        Redefine MEDIA_DIR in constants.py to point to 
        a valid directory where movies and images will 
        be written
    """)
for folder in [FILE_DIR, RASTER_IMAGE_DIR, SVG_IMAGE_DIR, ANIMATIONS_DIR, TEX_DIR,
               TEX_IMAGE_DIR, MOBJECT_DIR, IMAGE_MOBJECT_DIR,
               STAGED_SCENES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

TEX_TEXT_TO_REPLACE = "YourTextHere"
TEMPLATE_TEX_FILE  = os.path.join(THIS_DIR, "template.tex")
TEMPLATE_TEXT_FILE = os.path.join(THIS_DIR, "text_template.tex")


FFMPEG_BIN = "ffmpeg"


### Colors ###


COLOR_MAP = {
    "DARK_BLUE"   : "#236B8E",
    "DARK_BROWN"  : "#8B4513",
    "LIGHT_BROWN" : "#CD853F",
    "BLUE_E"      : "#1C758A",
    "BLUE_D"      : "#29ABCA",
    "BLUE_C"      : "#58C4DD",
    "BLUE_B"      : "#9CDCEB",
    "BLUE_A"      : "#C7E9F1",
    "TEAL_E"      : "#49A88F",
    "TEAL_D"      : "#55C1A7",
    "TEAL_C"      : "#5CD0B3",
    "TEAL_B"      : "#76DDC0",
    "TEAL_A"      : "#ACEAD7",
    "GREEN_E"     : "#699C52",
    "GREEN_D"     : "#77B05D",
    "GREEN_C"     : "#83C167",
    "GREEN_B"     : "#A6CF8C",
    "GREEN_A"     : "#C9E2AE",
    "YELLOW_E"    : "#E8C11C",
    "YELLOW_D"    : "#F4D345",
    "YELLOW_C"    : "#FFFF00",
    "YELLOW_B"    : "#FFEA94",
    "YELLOW_A"    : "#FFF1B6",
    "GOLD_E"      : "#C78D46",
    "GOLD_D"      : "#E1A158",
    "GOLD_C"      : "#F0AC5F",
    "GOLD_B"      : "#F9B775",
    "GOLD_A"      : "#F7C797",
    "RED_E"       : "#CF5044",
    "RED_D"       : "#E65A4C",
    "RED_C"       : "#FC6255",
    "RED_B"       : "#FF8080",
    "RED_A"       : "#F7A1A3",
    "MAROON_E"    : "#94424F",
    "MAROON_D"    : "#A24D61",
    "MAROON_C"    : "#C55F73",
    "MAROON_B"    : "#EC92AB",
    "MAROON_A"    : "#ECABC1",
    "PURPLE_E"    : "#644172",
    "PURPLE_D"    : "#715582",
    "PURPLE_C"    : "#9A72AC",
    "PURPLE_B"    : "#B189C6",
    "PURPLE_A"    : "#CAA3E8",
    "WHITE"       : "#FFFFFF",
    "BLACK"       : "#000000",
    "LIGHT_GRAY"  : "#BBBBBB",
    "LIGHT_GREY"  : "#BBBBBB",
    "GRAY"        : "#888888",
    "GREY"        : "#888888",
    "DARK_GREY"   : "#444444",
    "DARK_GRAY"   : "#444444",
    "GREY_BROWN"  : "#736357",
    "PINK"        : "#D147BD",
    "GREEN_SCREEN": "#00FF00",
    "ORANGE"      : "#FF862F",
}
PALETTE = COLOR_MAP.values()
# 将COLOR_MAP中的信息放到局部变量中去，当其它模块导入constants时也会导入这些变量信息。
locals().update(COLOR_MAP)
# 将以_C结尾的键返回，去掉_C并将原来的值赋值给它
for name in filter(lambda s : s.endswith("_C"), COLOR_MAP.keys()):
    locals()[name.replace("_C", "")] = locals()[name]










