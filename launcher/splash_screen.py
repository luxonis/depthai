# TODO(themarpe) - crossplatform and with capability of resizing splash image based on display size

# Windows specific way of creating a splash screen
import tkinter as tk
import sys, threading, time

# Event name to indicate a 'quit' request
QUIT_EVENT = '<<quit>>'

def create(image_file):
    splash_root = tk.Tk()
    splash_root.title("Splash")
    splash_root.overrideredirect(True)

    # The image must be stored to Tk or it will be garbage collected.
    splash_root.image = tk.PhotoImage(file=image_file)
    label = tk.Label(splash_root, image=splash_root.image, bg='white')
    splash_root.wm_attributes("-topmost", True)
    splash_root.wm_attributes("-disabled", True)
    splash_root.wm_attributes("-transparentcolor", "white")
    label.pack()

    imgw = splash_root.image.width()
    imgh = splash_root.image.height()
    screenw = splash_root.winfo_screenwidth()
    screenh = splash_root.winfo_screenheight()
    tlx = int((screenw - imgw) / 2)
    tly = int((screenh - imgh) / 2)
    splash_root.geometry(f'{imgw}x{imgh}+{tlx}+{tly}')

    #print(f'w: {imgw}, h: {imgh}, screen w: {screenw}, screen h: {screenh}')
    return splash_root

def quit(splash, timeout):
    time.sleep(timeout)
    splash.event_generate(QUIT_EVENT, when="tail")

# execute only if run as a script
if __name__ == "__main__":
    splashImage = 'splash2.png'

    timeout = 3
    if len(sys.argv) > 1:
        splashImage = sys.argv[1]
    if len(sys.argv) > 2:
        timeout = int(sys.argv[2])

    # Create splash screen
    splash = create(splashImage)

    # Create a timeout task (if timeout > 0)
    if timeout > 0:
        threading.Thread(target=quit, args=[splash]).start()
   
    # Add event '<<quit>>' to call quit on root splash screen
    splash.bind(QUIT_EVENT, (lambda splash: lambda args: splash.quit())(splash))

    # Run main loop
    splash.mainloop()
