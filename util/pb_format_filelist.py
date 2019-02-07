import os
import Tkinter as tk
import pdb

root = tk.Tk()
# keep the window from showing
root.withdraw()

def main():
	cbtxt = root.clipboard_get()
	filelist = [os.path.split(fp)[-1] for fp in cbtxt.split('\n')]
	pastetxt = '{\'%s\'};' % ('\',\''.join(filelist))
	root.clipboard_clear()
	root.clipboard_append(pastetxt)
	# pdb.set_trace()
	raw_input('Clipboard Ready. Press Enter to quit and clear clipboard...')
	print pastetxt

if __name__ == '__main__':
	main()
