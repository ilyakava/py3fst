import os

def mkdirp(path):
  if not os.path.exists(path):
    print('Creating directory: %s' % path)
    os.makedirs(path)