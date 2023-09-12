from __future__ import print_function

import os
import yaml
import time
from IPython.display import Image, display
from IPython.core.magic import *

def _display_image(path):
    print(path)
    display(Image(path))

def wait_and_show_image(path, sleep=2, loop=False, show_all=False):
    try:
        last = ''
        path = os.path.expanduser(path)
        while loop or last == '':
            imgs = [ "{}/{}".format(path, i) for i in os.listdir(path) if 'png' in i or 'jpg' in i  ]
            imgs = sorted(imgs, key=lambda x: os.path.getmtime(x))[::-1]
            if len(imgs) == 0:
                print("No image found")
                break
            if show_all:
                for i in imgs:
                    _display_image(i)
            elif imgs[0] != last:
                _display_image(imgs[0])
                last = imgs[0]
            time.sleep(sleep)
    except KeyboardInterrupt:
        print("Done!")
