"""Movable X11 capture frame with a genuinely empty center."""
import ctypes as C
import ctypes.util
import tkinter as tk


class Rectangle(C.Structure):
    _fields_ = [("x", C.c_short), ("y", C.c_short),
                ("width", C.c_ushort), ("height", C.c_ushort)]


class CaptureRegion:
    def __init__(self, root, on_close):
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.geometry("640x400+200+400")
        self.canvas = tk.Canvas(self.window, bg="#21b6a8", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.xlib = C.CDLL(ctypes.util.find_library("X11"))
        self.shape = C.CDLL(ctypes.util.find_library("Xext"))
        self.xlib.XOpenDisplay.argtypes = [C.c_char_p]
        self.xlib.XOpenDisplay.restype = C.c_void_p
        self.xlib.XFlush.argtypes = [C.c_void_p]
        self.xlib.XCloseDisplay.argtypes = [C.c_void_p]
        self.xlib.XQueryTree.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ulong)), C.POINTER(C.c_uint)]
        self.xlib.XFree.argtypes = [C.c_void_p]
        self.shape.XShapeCombineRectangles.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_int, C.c_int, C.POINTER(Rectangle), C.c_int, C.c_int, C.c_int]
        self.display = self.xlib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("The transparent capture frame requires an X11 desktop.")
        self.window.bind("<Configure>", self.reshape)
        self.canvas.bind("<ButtonPress-1>", self.begin_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.window.bind("<Escape>", on_close)
        self.window.update_idletasks()
        self.reshape()

    def reshape(self, event=None):
        w, h = self.window.winfo_width(), self.window.winfo_height()
        if w < 10 or h < 33:
            return
        rectangles = (Rectangle * 5)(Rectangle(0, 0, w, 28), Rectangle(0, 28, 5, h-28), Rectangle(w-5, 28, 5, h-28), Rectangle(0, h-5, w, 5), Rectangle(w-20, h-20, 20, 20))
        client = self.window.winfo_id()
        root, parent = C.c_ulong(), C.c_ulong()
        children, count = C.POINTER(C.c_ulong)(), C.c_uint()
        self.xlib.XQueryTree(self.display, client, C.byref(root), C.byref(parent), C.byref(children), C.byref(count))
        if children:
            self.xlib.XFree(children)
        for xid in {client, parent.value}:
            if xid and xid != root.value:
                self.shape.XShapeCombineRectangles(self.display, xid, 0, 0, 0, rectangles, 5, 0, 0)
        self.xlib.XFlush(self.display)
        self.canvas.delete("all")
        self.canvas.create_text(12, 14, anchor="w", text="Drag to move | Sound samples inside this frame | ESC: quit", fill="black")
        self.canvas.create_rectangle(w-20, h-20, w, h, fill="#21b6a8", outline="")

    def begin_drag(self, event):
        self.start = (event.x_root, event.y_root, self.window.winfo_x(), self.window.winfo_y(), self.window.winfo_width(), self.window.winfo_height())
        self.resizing = event.x > self.window.winfo_width()-24 and event.y > self.window.winfo_height()-24

    def drag(self, event):
        sx, sy, x, y, w, h = self.start
        dx, dy = event.x_root-sx, event.y_root-sy
        if self.resizing:
            self.window.geometry(f"{max(360, w+dx)}x{max(120, h+dy)}+{x}+{y}")
        else:
            self.window.geometry(f"+{max(0, x+dx)}+{max(0, y+dy)}")

    def bbox(self):
        x, y = self.window.winfo_rootx(), self.window.winfo_rooty()
        return (x+5, y+28, x+self.window.winfo_width()-5, y+self.window.winfo_height()-5)

    def close(self):
        self.xlib.XCloseDisplay(self.display)
