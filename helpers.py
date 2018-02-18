# _*_ coding:utf-8 _*_

import numpy as np
import itertools as it
import operator as op
from PIL import Image
from colour import Color
import random
import inspect
import string
import re
import os
from scipy import linalg

from constants import *

CLOSED_THRESHOLD = 0.01
STRAIGHT_PATH_THRESHOLD = 0.01

# 定义一些列的帮助函数
# chord 弦乐
# 这里构造的是在ubuntu下的一个命令用来播放声音
def play_chord(*nums):
    commands = [
        "play",
        "-n",
        "-c1",
        "--no-show-progress",
        "synth",
    ] + [
        "sin %-"+str(num)
        for num in nums
    ] + [
        "fade h 0.5 1 0.5", 
        "> /dev/null"
    ]
    try:
        os.system(" ".join(commands))
    except:
        pass

# 播放失败的声音
def play_error_sound():
    play_chord(11, 8, 6, 1)

# 播放结束的声音
def play_finish_sound():
    play_chord(12, 9, 5, 2)

# smooth spline 平滑逼近
def get_smooth_handle_points(points):
    points = np.array(points)
    num_handles = len(points) - 1
    # 获取返回的shape元组中第二个元素
    dim = points.shape[1]    
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    #Must solve 2*num_handles equations to get the handles.
    #l and u are the number of lower an upper diagonal(对角) rows
    #in the matrix to solve.
    l, u = 2, 1    
    #diag is a representation of the matrix in diagonal form
    #See https://www.particleincell.com/2012/bezier-splines/
    #for how to arive at these equations
    diag = np.zeros((l+u+1, 2*num_handles))
    diag[0,1::2] = -1
    diag[0,2::2] = 1
    diag[1,0::2] = 2
    diag[1,1::2] = 1
    diag[2,1:-2:2] = -2
    diag[3,0:-3:2] = 1
    #last
    diag[2,-2] = -1
    diag[1,-1] = 2
    #This is the b as in Ax = b, where we are solving for x,
    #and A is represented using diag.  However, think of entries
    #to x and b as being points in space, not numbers
    b = np.zeros((2*num_handles, dim))
    b[1::2] = 2*points[1:]
    b[0] = points[0]
    b[-1] = points[-1]
    solve_func = lambda b : linalg.solve_banded(
        (l, u), diag, b
    )
    if is_closed(points):
        #Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        #last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        #first row handles first derivative
        matrix[0,:] = np.zeros(matrix.shape[1])
        matrix[0,[0, -1]] = [1, 1]
        b[0] = 2*points[0]
        b[-1] = np.zeros(dim)
        solve_func = lambda b : linalg.solve(matrix, b)
    handle_pairs = np.zeros((2*num_handles, dim))
    for i in range(dim):
        handle_pairs[:,i] = solve_func(b[:,i])
    return handle_pairs[0::2], handle_pairs[1::2]

def diag_to_matrix(l_and_u, diag):
    """
    Converts array whose rows represent diagonal 
    entries of a matrix into the matrix itself.
    See scipy.linalg.solve_banded
    """
    l, u = l_and_u
    dim = diag.shape[1]
    matrix = np.zeros((dim, dim))
    for i in range(l+u+1):
        np.fill_diagonal(
            matrix[max(0,i-u):,max(0,u-i):],
            diag[i,max(0,u-i):]
        )
    return matrix

# 对第一个点和最后一个点的差求二范数,然后和阈值进行判断
def is_closed(points):
    return np.linalg.norm(points[0] - points[-1]) < CLOSED_THRESHOLD

## Color
# 将颜色从constants文件中的"#236B8E"形式转化为rgb格式
def color_to_rgb(color):
    return np.array(Color(color).get_rgb())

#rgba其中a这个参数表示alpha 透明度，默认是1就是不透明
def color_to_rgba(color, alpha = 1):
    return np.append(color_to_rgb(color), [alpha])

#将rgb转化为颜色对象
def rgb_to_color(rgb):
    try:
        return Color(rgb = rgb)
    except:
        return Color(WHITE)

#将rgba转化为颜色对象，不使用透明度参数
def rgba_to_color(rgba):
    return rgb_to_color(rgba[:3])

#转为16进制
def rgb_to_hex(rgb):
    return Color(rgb = rgb).get_hex_l()

#翻转颜色
def invert_color(color):
    return rgb_to_color(1.0 - color_to_rgb(color))

#将颜色转化为以255来计算的整型数据
def color_to_int_rgb(color):
    return (255*color_to_rgb(color)).astype('uint8')

#将rgba颜色转化以255计算的整数型数据
def color_to_int_rgba(color, alpha = 255):
    return np.append(color_to_int_rgb(color), alpha)

#制作颜色梯度，输入的reference_colors的颜色包含来起始颜色和结束颜色
def color_gradient(reference_colors, length_of_output):
    if length_of_output == 0:
        return reference_colors[0]
    # 转化为rgb表示的颜色
    rgbs = map(color_to_rgb, reference_colors)
    # 设置梯度
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype('int')
    alphas_mod1 = alphas % 1
    #End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color(interpolate(rgbs[i], rgbs[i+1], alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]

# 插入颜色，返回在两个颜色之间的颜色
def interpolate_color(color1, color2, alpha):
    rgb = interpolate(color_to_rgb(color1), color_to_rgb(color2), alpha)
    return rgb_to_color(rgb)

def average_color(*colors):
    rgbs = np.array(map(color_to_rgb, colors))
    #numpy.apply_along_axis(func, axis, arr, *args, **kwargs)
    #func 应用函数，axis轴 ，arr被应用的函数
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)

###

def compass_directions(n = 4, start_vect = RIGHT):
    angle = 2*np.pi/n
    return np.array([
        # 旋转向量
        rotate_vector(start_vect, k*angle)
        for k in range(n)
    ])

#bezier curve贝塞尔曲线，这个曲线是用来在计算机上绘制曲线
#的基础。

def partial_bezier_points(points, a, b):
    """
    Given an array of points which define 
    a bezier curve, and two numbers 0<=a<b<=1,
    return an array of the same size, which 
    describes the portion of the original bezier
    curve on the interval [a, b].

    This algorithm is pretty nifty, and pretty dense.
    """
    a_to_1 = np.array([
        bezier(points[i:])(a)
        for i in range(len(points))
    ])
    return np.array([
        bezier(a_to_1[:i+1])((b-a)/(1.-a))
        for i in range(len(points))
    ])

def bezier(points):
    n = len(points) - 1
    return lambda t : sum([
        ((1-t)**(n-k))*(t**k)*choose(n, k)*point
        for k, point in enumerate(points)
    ])

# 通过set只能放置唯一的元素的特点，对list进行去重
def remove_list_redundancies(l):
    """
    Used instead of list(set(l)) to maintain order
    """
    result = []
    used = set()
    for x in l:
        if not x in used:
            result.append(x)
            used.add(x)
    return result

# 将被更新对象l1中和l2重复的部分剔除，然后将l2加在l1中
def list_update(l1, l2):
    """
    Used instead of list(set(l1).update(l2)) to maintain order,
    making sure duplicates are removed from l1, not l2.
    """
    return filter(lambda e : e not in l2, l1) + list(l2)

#将不在l2中的l1元素过滤出来
def list_difference_update(l1, l2):
    return filter(lambda e : e not in l2, l1)

#检查是否所有的对象都属于某个类
def all_elements_are_instances(iterable, Class):
    return all(map(lambda e : isinstance(e, Class), iterable))

#将一个对象和他相邻的对象成对的返回
def adjacent_pairs(objects):
    return zip(objects, list(objects[1:])+[objects[0]])

#将复数的实数部分和虚数部分分别保存numpy的array中
def complex_to_R3(complex_num):
    return np.array((complex_num.real, complex_num.imag, 0))

#将点转化为复数
def R3_to_complex(point):
    return complex(*point[:2])

# 元组化
def tuplify(obj):
    if isinstance(obj, str):
        return (obj,)
    try:
        return tuple(obj)
    except:
        return (obj,)

def instantiate(obj):
    """
    Useful so that classes or instance of those classes can be 
    included in configuration, which can prevent defaults from
    getting created during compilation/importing
    """
    return obj() if isinstance(obj, type) else obj

# descendent 后裔，派生类
# 循环的获取所有的派生类
def get_all_descendent_classes(Class):
    awaiting_review = [Class]
    result = []
    while awaiting_review:
        Child = awaiting_review.pop()
        awaiting_review += Child.__subclasses__()
        result.append(Child)
    return result

# 过滤 self 和 kwargs 参数
def filtered_locals(caller_locals):
    result = caller_locals.copy()
    ignored_local_args = ["self", "kwargs"]
    for arg in ignored_local_args:
        result.pop(arg, caller_locals)
    return result

# 将CONFIG中的信息转到类中成为成员变量
def digest_config(obj, kwargs, caller_locals = {}):
    """
    Sets init args and CONFIG values as local variables

    The purpose of this function is to ensure that all 
    configuration of any object is inheritable, able to 
    be easily passed into instantiation, and is attached
    as an attribute of the object.
    """

    # Assemble list of CONFIGs from all super classes
    classes_in_hierarchy = [obj.__class__]
    static_configs = []
    # 将该类以及其父类中所有的CONFIG都放置到统一的位置
    while len(classes_in_hierarchy) > 0:
        Class = classes_in_hierarchy.pop()
        classes_in_hierarchy += Class.__bases__
        if hasattr(Class, "CONFIG"):
            static_configs.append(Class.CONFIG)

    #Order matters a lot here, first dicts have higher priority
    caller_locals = filtered_locals(caller_locals)
    all_dicts = [kwargs, caller_locals, obj.__dict__]
    all_dicts += static_configs
    all_new_dicts = [kwargs, caller_locals] + static_configs
    obj.__dict__ = merge_config(all_dicts)
    #Keep track of the configuration of objects upon 
    #instantiation
    obj.initial_config = merge_config(all_new_dicts)

# 合并配置
def merge_config(all_dicts):
    all_config = reduce(op.add, [d.items() for d in all_dicts])
    config = dict()
    for c in all_config:
        key, value = c
        if not key in config:
            config[key] = value
        else:
            #When two dictionaries have the same key, they are merged.
            if isinstance(value, dict) and isinstance(config[key], dict):
                config[key] = merge_config([config[key], value])
    return config

def soft_dict_update(d1, d2):
    """
    Adds key values pairs of d2 to d1 only when d1 doesn't
    already have that key
    """
    for key, value in d2.items():
        if key not in d1:
            d1[key] = value
# 消化本地变量
def digest_locals(obj, keys = None):
    caller_locals = filtered_locals(
        inspect.currentframe().f_back.f_locals
    )
    if keys is None:
        keys = caller_locals.keys()
    for key in keys:
        setattr(obj, key, caller_locals[key])

# 插入
def interpolate(start, end, alpha):
    return (1-alpha)*start + alpha*end

# 中间值
def mid(start, end):
    return (start + end)/2.0

# 反插入 np.true_divide 真除法，返回的是浮点数
# 返回用于插入时的alpha值
def inverse_interpolate(start, end, value):
    return np.true_divide(value - start, end - start)

# 设置val的最大最小阈值
def clamp(lower, upper, val):
    if val < lower:
        return lower
    elif val > upper:
        return upper
    return val

#将点转化为浮点型，并计算点的平均值
def center_of_mass(points):
    points = [np.array(point).astype("float") for point in points]
    return sum(points) / len(points)

#以r来划分n，将1到r的乘作为分母，r到n作为分子，将他们的商返回。
def choose(n, r):
    if n < r: return 0
    if r == 0: return 1
    denom = reduce(op.mul, xrange(1, r+1), 1)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    return numer//denom

#判断p0是否在p1和p2所在的直线上
def is_on_line(p0, p1, p2, threshold = 0.01):
    """
    Returns true of p0 is on the line between p1 and p2
    """
    p0, p1, p2 = map(lambda tup : np.array(tup[:2]), [p0, p1, p2])
    p1 -= p0
    p2 -= p0
    return abs((p1[0] / p1[1]) - (p2[0] / p2[1])) < threshold

# 交点
def intersection(line1, line2):
    """
    A "line" should come in the form [(x0, y0), (x1, y1)] for two
    points it runs through
    """
    p0, p1, p2, p3 = map(
        lambda tup : np.array(tup[:2]),
        [line1[0], line1[1], line2[0], line2[1]]
    )
    p1, p2, p3 = map(lambda x : x - p0, [p1, p2, p3])
    transform = np.zeros((2, 2))
    transform[:,0], transform[:,1] = p1, p2
    if np.linalg.det(transform) == 0: return
    inv = np.linalg.inv(transform)
    new_p3 = np.dot(inv, p3.reshape((2, 1)))
    #Where does line connecting (0, 1) to new_p3 hit x axis
    x_intercept = new_p3[0] / (1 - new_p3[1]) 
    result = np.dot(transform, [[x_intercept], [0]])
    result = result.reshape((2,)) + p0
    return result

#随机产生一个比较亮的颜色
def random_bright_color():
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate(
        curr_rgb, np.ones(len(curr_rgb)), 0.5
    )
    return Color(rgb = new_rgb)

# 随机选择一个颜色 random 的 choice
def random_color():
    return random.choice(PALETTE)


################################################
#直线路径
def straight_path(start_points, end_points, alpha):
    return interpolate(start_points, end_points, alpha)

#弧形路径
def path_along_arc(arc_angle, axis = OUT):
    """
    If vect is vector from start to end, [vect[:,1], -vect[:,0]] is 
    perpendicular to vect in the left direction.
    """
    if abs(arc_angle) < STRAIGHT_PATH_THRESHOLD:
        return straight_path
    if np.linalg.norm(axis) == 0:
        axis = OUT
    unit_axis = axis/np.linalg.norm(axis)
    def path(start_points, end_points, alpha):
        vects = end_points - start_points
        centers = start_points + 0.5*vects
        if arc_angle != np.pi:
            centers += np.cross(unit_axis, vects/2.0)/np.tan(arc_angle/2)
        rot_matrix = rotation_matrix(alpha*arc_angle, unit_axis)
        return centers + np.dot(start_points-centers, rot_matrix.T)
    return path

#顺时针路径
def clockwise_path():
    return path_along_arc(-np.pi)

#逆时针路径
def counterclockwise_path():
    return path_along_arc(np.pi)


################################################

def to_camel_case(name):
    return "".join([
        filter(
            lambda c : c not in string.punctuation + string.whitespace, part
        ).capitalize()
        for part in name.split("_")
    ])

def initials(name, sep_values = [" ", "_"]):
    return "".join([
        (s[0] if s else "") 
        for s in re.split("|".join(sep_values), name)
    ])

def camel_case_initials(name):
    return filter(lambda c : c.isupper(), name)

################################################

def get_full_raster_image_path(image_file_name):
    possible_paths = [
        image_file_name,
        os.path.join(RASTER_IMAGE_DIR, image_file_name),
        os.path.join(RASTER_IMAGE_DIR, image_file_name + ".jpg"),
        os.path.join(RASTER_IMAGE_DIR, image_file_name + ".png"),
        os.path.join(RASTER_IMAGE_DIR, image_file_name + ".gif"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise IOError("File %s not Found"%image_file_name)

def drag_pixels(frames):
    curr = frames[0]
    new_frames = []
    for frame in frames:
        curr += (curr == 0) * np.array(frame)
        new_frames.append(np.array(curr))
    return new_frames

#翻转图片
def invert_image(image):
    arr = np.array(image)
    arr = (255 * np.ones(arr.shape)).astype(arr.dtype) - arr
    return Image.fromarray(arr)

def stretch_array_to_length(nparray, length):
    curr_len = len(nparray)
    if curr_len > length:
        raise Warning("Trying to stretch array to a length shorter than its own")
    indices = np.arange(length)/ float(length)
    indices *= curr_len
    return nparray[indices.astype('int')]

#
def make_even(iterable_1, iterable_2):
    list_1, list_2 = list(iterable_1), list(iterable_2)
    length = max(len(list_1), len(list_2))
    return (
        [list_1[(n * len(list_1)) / length] for n in xrange(length)],
        [list_2[(n * len(list_2)) / length] for n in xrange(length)]
    )

def make_even_by_cycling(iterable_1, iterable_2):
    length = max(len(iterable_1), len(iterable_2))
    cycle1 = it.cycle(iterable_1)
    cycle2 = it.cycle(iterable_2)
    return (
        [cycle1.next() for x in range(length)],
        [cycle2.next() for x in range(length)]
    )

### Rate Functions ###

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def smooth(t, inflection = 10.0):
    error = sigmoid(-inflection / 2)
    return (sigmoid(inflection*(t - 0.5)) - error) / (1 - 2*error)

def rush_into(t):
    return 2*smooth(t/2.0)

def rush_from(t):
    return 2*smooth(t/2.0+0.5) - 1

def slow_into(t):
    return np.sqrt(1-(1-t)*(1-t))

def double_smooth(t):
    if t < 0.5:
        return 0.5*smooth(2*t)
    else:
        return 0.5*(1 + smooth(2*t - 1))

def there_and_back(t, inflection = 10.0):
    new_t = 2*t if t < 0.5 else 2*(1 - t)
    return smooth(new_t, inflection)

def there_and_back_with_pause(t):
    if t < 1./3:
        return smooth(3*t)
    elif t < 2./3:
        return 1
    else:
        return smooth(3 - 3*t)

def running_start(t, pull_factor = -0.5):
    return bezier([0, 0, pull_factor, pull_factor, 1, 1, 1])(t)

def not_quite_there(func = smooth, proportion = 0.7):
    def result(t):
        return proportion*func(t)
    return result

def wiggle(t, wiggles = 2):
    return there_and_back(t) * np.sin(wiggles*np.pi*t)

def squish_rate_func(func, a = 0.4, b = 0.6):
    def result(t):
        if t < a:
            return func(0)
        elif t > b:
            return func(1)
        else:
            return func((t-a)/(b-a))
    return result

# Stylistically, should this take parameters (with default values)?
# Ultimately, the functionality is entirely subsumed by squish_rate_func,
# but it may be useful to have a nice name for with nice default params for 
# "lingering", different from squish_rate_func's default params
def lingering(t):
    return squish_rate_func(lambda t: t, 0, 0.8)(t)

### Functional Functions ###

def composition(func_list):
    """
    func_list should contain elements of the form (f, args)
    """
    return reduce(
        lambda (f1, args1), (f2, args2) : (lambda x : f1(f2(x, *args2), *args1)), 
        func_list,
        lambda x : x
    )

def remove_nones(sequence):
    return filter(lambda x : x, sequence)

#Matrix operations
def thick_diagonal(dim, thickness = 2):
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    col_indices = np.transpose(row_indices)
    return (np.abs(row_indices - col_indices)<thickness).astype('uint8')
# 查看矩阵的旋转可以看看这个博客
#http://blog.csdn.net/csxiaoshui/article/details/65446125
def rotation_matrix(angle, axis):
    """
    Rotation in R^3 about a specified axis of rotation.
    """
    about_z = rotation_about_z(angle)
    z_to_axis = z_to_vector(axis)
    # 矩阵求逆np.linalg.inv
    axis_to_z = np.linalg.inv(z_to_axis)
    return reduce(np.dot, [z_to_axis, about_z, axis_to_z])

def rotation_about_z(angle):
    return [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ]

def z_to_vector(vector):
    """
    Returns some matrix in SO(3) which takes the z-axis to the 
    (normalized) vector provided as an argument
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.identity(3)
    v = np.array(vector) / norm
    phi = np.arccos(v[2])
    if any(v[:2]):
        #projection of vector to unit circle
        axis_proj = v[:2] / np.linalg.norm(v[:2])
        theta = np.arccos(axis_proj[0])
        if axis_proj[1] < 0:
            theta = -theta
    else:
        theta = 0
    phi_down = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    return np.dot(rotation_about_z(theta), phi_down)

def rotate_vector(vector, angle, axis = OUT):
    return np.dot(rotation_matrix(angle, axis), vector)

def angle_between(v1, v2):
    return np.arccos(np.dot(
        v1 / np.linalg.norm(v1), 
        v2 / np.linalg.norm(v2)
    ))

def angle_of_vector(vector):
    """
    Returns polar coordinate theta when vector is project on xy plane
    """
    # complex 复数 选择前两个数字进行复数计算如果是0则直接返回0
    # 否则将数值转换为角度
    z = complex(*vector[:2])
    if z == 0:
        return 0
    return np.angle(complex(*vector[:2]))

def concatenate_lists(*list_of_lists):
    return [item for l in list_of_lists for item in l]

# Occasionally convenient in order to write dict.x instead of more laborious 
# (and less in keeping with all other attr accesses) dict["x"]
class DictAsObject(object):
    def __init__(self, dict):
         self.__dict__ = dict

# Just to have a less heavyweight name for this extremely common operation
def fdiv(a, b):
    return np.true_divide(a,b)
