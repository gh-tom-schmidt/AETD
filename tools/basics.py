class DirectionOutOfBounds(Exception):
    def __init__(self, direction: int):
        super().__init__(f"Direction must be between -1, 0 or 1, yours is: {direction}")


class SpeedOutOfBounds(Exception):
    def __init__(self, speed: int):
        super().__init__(f"The speed must be between 0 and 100, yours is: {speed}")
