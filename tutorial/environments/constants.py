from enum import IntEnum, Enum

class Directions(IntEnum):
	right 	= 0 
	down 	= 1
	left 	= 2
	up 		= 3

class DirectionsXY(Enum): 
	# reference: minigrid.core.constants
	right 	= (1, 0)
	down 	= (0, 1)
	left    = (-1, 0)
	up 	    = (0, -1)
