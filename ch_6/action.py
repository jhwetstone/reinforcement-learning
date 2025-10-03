from enum import Enum

class CoordinateEnum(Enum):
    '''Base enum class with x,y coordinate attributes'''

    def __new__(cls, display_name, x_coord, y_coord):
        obj = object.__new__(cls)
        obj._value_ = display_name  # The primary value of the enum member
        obj.x = x_coord            # Custom attribute for x-coordinate
        obj.y = y_coord            # Custom attribute for y-coordinate
        return obj

class Action(CoordinateEnum):
    '''Original set of actions: Up, Down, Right, Left'''
    UP = (0, 0, 1)
    DOWN = (1, 0, -1)
    RIGHT = (2, 1, 0)
    LEFT = (3, -1, 0)

class KingAction(CoordinateEnum):
    '''Includes diagonal actions as well'''
    UP = (0, 0, 1)
    DOWN = (1, 0, -1)
    RIGHT = (2, 1, 0)
    LEFT = (3, -1, 0)
    UP_RIGHT = (4, 1, 1)
    DOWN_RIGHT = (5, 1, -1)
    UP_LEFT = (6, -1, 1)
    DOWN_LEFT = (7, -1, -1)

class KingActionWithStayOption(CoordinateEnum):
    '''Includes diagonal actions + a stay option'''
    UP = (0, 0, 1)
    DOWN = (1, 0, -1)
    RIGHT = (2, 1, 0)
    LEFT = (3, -1, 0)
    UP_RIGHT = (4, 1, 1)
    DOWN_RIGHT = (5, 1, -1)
    UP_LEFT = (6, -1, 1)
    DOWN_LEFT = (7, -1, -1)
    STAY = (8, 0, 0)