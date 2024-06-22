from calibrate import Main
import pathlib

device = "OAK-D-PRO"
save_folder = str(pathlib.Path(__file__).resolve().parent)
print(save_folder)
static = ['-s', '5.0', '-nx', '29', '-ny', '29', '-ms', '3.7', '-dbg', '-m', 'process', '-brd', device + ".json", "-scp", save_folder]
left_binary = "010100000"
right_binary = "010100000"
color_binary = "010100000"
dynamic = static + ['-pccm', 'left=' + left_binary, 'right=' + right_binary, "rgb=" + color_binary]
main = Main(dynamic)
main.run()