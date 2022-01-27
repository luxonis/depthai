from depthai_helpers.pygame_checkbox import Checkbox, pygame_render_text
import pygame
from pygame.locals import *
import os

WHITE  = (255, 255, 255)
ORANGE = (143, 122, 4)
RED    = (230, 9, 9)
GREEN  = (4, 143, 7)
BLACK  = (0, 0, 0)

pygame.init()
WIDTH, HEIGHT = 900, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
test_type = 'OAK_D' # TODO get device type?
pygame.display.set_caption(test_type)

FPS = 60
PASS_IMAGE = pygame.image.load(os.path.join('Assets', 'pass.PNG'))
PASS_OPTION = pygame.transform.scale(PASS_IMAGE, (55, 40))
FONT = pygame.font.Font(None, 20)

def draw_window():
    WIN.fill(WHITE)
    title = "UNIT TEST IN PROGRESS"
    pygame_render_text(WIN, title, (200, 20), ORANGE, 50)

    heading = "Automated Tests                  Operator Tests"
    pygame_render_text(WIN, heading, (249, 70), BLACK, 30)
    WIN.blit(PASS_OPTION, (300, 100))
    if test_type == 'OAK_D':
        auto_checkbox_names = ["USB3", "Left camera connected", "Right camera connected",
                                       "RGB camera connected", "JPEG Encoding Stream",
                                       "previewout-rgb Stream", "left Stream", "right Stream"]
        auto_checkbox_names.append('IMU')
        op_checkbox_names = ["JPEG Encoding", "Previewout-rgb stream", "Left Stream", "Right Stream"]

    auto_checkbox_dict = {}
    x = 200
    y = 110
    for i in range(len(auto_checkbox_names)):
        w, h = FONT.size(auto_checkbox_names[i])
        x_axis = x - w
        y_axis = y +  (40*i)
        font_surf = FONT.render(auto_checkbox_names[i], True, GREEN)
        WIN.blit(font_surf, (x_axis,y_axis))
        auto_checkbox_dict[auto_checkbox_names[i]] = Checkbox(WIN, x + 10, y_axis-5, outline_color=GREEN,
                                                    check_color=GREEN)
    y = 150
    x = 550
    heading = "PASS"
    pygame_render_text(WIN, heading, (x, y - 30), GREEN, 30)
    heading = "Not"
    pygame_render_text(WIN, heading, (x + 70, y - 50), ORANGE, 30)

    heading = "Tested"
    pygame_render_text(WIN, heading, (x + 60, y - 30), ORANGE, 30)
    heading = "FAIL"
    pygame_render_text(WIN, heading, (x + 150, y - 30), RED, 30)

    op_checkbox_dict = {}
    for i in range(len(op_checkbox_names)):
        w, h = FONT.size(op_checkbox_names[i])
        x_axis = x - w
        y_axis = y +  (40*i)
        font_surf = FONT.render(op_checkbox_names[i], True, ORANGE)
        WIN.blit(font_surf, (x_axis,y_axis))
        checker_list = [Checkbox(WIN, x + 10, y_axis-5, outline_color=GREEN, check_color=GREEN, disable_pass = True), 
                        Checkbox(WIN, x + 80, y_axis-5, outline_color=ORANGE, check_color=ORANGE, disable_pass = True, check=True),
                        Checkbox(WIN, x + (80*2), y_axis-5, outline_color=RED, check_color=RED, disable_pass = True)]
        op_checkbox_dict[op_checkbox_names[i]] = checker_list

    y = 380-50
    x = 330+200

    action_checkbox_dict = {}
    """
    for i in range(len(action_checkbox_names)):
        w, h = FONT.size(action_checkbox_names[i])
        x_axis = x - w
        y_axis = y +  (40*i)
        font_surf = FONT.render(action_checkbox_names[i], True, GREEN)
        WIN.blit(font_surf, (x_axis,y_axis))
        action_checkbox_dict[action_checkbox_names[i]] = Checkbox(WIN, x + 10, y_axis-5, outline_color=GREEN, 
                                                    check_color=ORANGE,
                                                    text_checked   = "ON",
                                                    text_unchecked = "OFF",
                                                    text_untested  = "FAIL")
    """
    # adding save button
    save_button =  pygame.Rect(600, 430, 60, 35)
    pygame.draw.rect(WIN, ORANGE, save_button)
    pygame_render_text(WIN, 'SAVE', (605, 440))
    is_saved = False
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_window()

    pygame.quit()

if __name__ == '__main__':
    main()

