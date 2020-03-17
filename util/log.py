import datetime

def write_metadata(logs_root, config):
  with open(('%s/metalog.txt' % 
             (logs_root)), 'w') as writefile:
    writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
    writefile.write('config: %s\n' % str(config))