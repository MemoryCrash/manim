# _*_ coding:utf-8 _*_

from helpers import *

# Currently, this is only used by both Scene and MOBject.
# Still, we abstract its functionality here, albeit purely nominally.
# All actual implementation has to be handled by derived classes for now.
#
# Note that although the prototypical instances add and remove MObjects, 
# there is also the possibility to add ContinualAnimations to Scenes. Thus, 
# in the Container class in general, we do not make any presumptions about 
# what types of objects may be added; this is again dependent on the specific
# derived instance.

#这个是个很基础的类。比较重要的是它在init函数中包含了digest_config函数
class Container(object):
    def __init__(self, *submobjects, **kwargs):
        digest_config(self, kwargs)

    def add(self, *items):
    	raise Exception("Container.add is not implemented; it is up to derived classes to implement")

    def remove(self, *items):
    	raise Exception("Container.remove is not implemented; it is up to derived classes to implement")