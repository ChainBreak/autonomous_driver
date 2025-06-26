import numpy as np
from typing import Any


class ActionCategorizer:
    """Converts a boolean action vector to a category number and back.
    """
    def __init__(self, action_vector_length:int):
        self.action_vector_length = action_vector_length
        self.binary_powers = 2 ** np.arange(action_vector_length)
        self.num_categories = 2 ** action_vector_length
        
    def to_category(self, action:np.ndarray[Any, np.dtype[np.bool_]]):
        """Convert action vector to category number"""
        return np.sum(action * self.binary_powers)
        
    def to_action(self, category:int):
        """Convert category number back to action vector"""
        # The & operator performs a bitwise AND between each bit of the category number
        # and the corresponding power of 2 (binary_powers). This checks which bits are set to 1.
        # For example:
        #   category=5 (binary 101) & binary_powers=[1,2,4] (binary [001,010,100])
        #   = [1,0,4] which becomes [True,False,True] after comparing >0
        return (category & self.binary_powers) > 0