import math
import random


class Cylinder:
    height = 0
    radius = 0

    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    def GetVolume(self):
        return math.pi * self.height * self.radius * self.radius

    def PrintInfo(self):
        print(str(self.radius) + ", " + str(self.height))



objs = []


def calc(func):
    from time import time

    def wrapper(*args, **kwargs):
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        print(f'用时: {end - start}s')
        return func_return

    return wrapper


@calc
def fun():
    while objs.__len__() > 0:
        r = random.randint(0, 9)
        for w in objs:
            if w["number"] == r:
                objs.remove(w)
                break


if __name__ == '__main__':
    for i in range(10):
        objs.append({
            "c": Cylinder(1, 1),
            "number": i
        })

    fun()
