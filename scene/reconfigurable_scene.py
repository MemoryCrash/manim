# _*_ coding:utf-8 _*_

import numpy as np

from scene import Scene
from animation.transform import Transform
from mobject import Mobject

from helpers import *

#Reconfigurable 可重构的
class ReconfigurableScene(Scene):
    CONFIG = {
        "allow_recursion" : True,
    }
    def setup(self):
        self.states = []
        # recursions 递归
        self.num_recursions = 0

    def transition_to_alt_config(
        self, 
        return_to_original_configuration = True, 
        transformation_kwargs = None,
        **new_config
        ):
        if transformation_kwargs is None:
            transformation_kwargs = {}
        original_state = self.get_state()
        state_copy = original_state.copy()
        self.states.append(state_copy)
        if not self.allow_recursion:
            return
        alt_scene = self.__class__(
            skip_animations = True, 
            allow_recursion = False,
            **new_config
        )
        alt_state = alt_scene.states[len(self.states)-1]

        if return_to_original_configuration:
            self.clear()
            self.transition_between_states(
                state_copy, alt_state, 
                **transformation_kwargs
            )
            self.transition_between_states(
                state_copy, original_state, 
                **transformation_kwargs
            )
            self.clear()
            self.add(*original_state)
        else:
            self.transition_between_states(
                original_state, alt_state, 
                **transformation_kwargs
            )
            self.__dict__.update(new_config)

    def get_state(self):
        # Want to return a mobject that maintains the most 
        # structure.  The way to do that is to extract only
        # those that aren't inside another.
        top_level_mobjects = self.get_top_level_mobjects()
        return Mobject(*self.get_top_level_mobjects())

    def transition_between_states(self, start_state, target_state, **kwargs):
        self.play(Transform(start_state, target_state, **kwargs))
        self.wait()



