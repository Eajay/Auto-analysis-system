import ctypes
from tkinter import *
from interface import Interface
if __name__ == "__main__":
    if 'win' in sys.platform:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.geometry("2600x1300")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(3, weight=1)
    analysis = Interface(master=root)
    root.mainloop()