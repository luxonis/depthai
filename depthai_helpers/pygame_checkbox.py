import pygame as pg

white  = [255, 255, 255]
orange = [143, 122, 4]
red    = [230, 9, 9]
green  = [4, 143, 7]
black  = [0, 0, 0]

def pygame_render_text(surface, text, pose, color=(0,0,0), font_size = 30 ):
    font = pg.font.Font(None, font_size)
    font_surf = font.render(text, True, color)
    surface.blit(font_surf, pose)

class Checkbox:
    def __init__(self, surface, x, y, color=(230, 230, 230), caption="", outline_color=(0, 0, 0),
                 check_color=(0, 0, 0), font_size=30, font_color=(0, 0, 0), text_offset=(28, 1),
                disable_pass = False, check = False):
        self.surface = surface
        self.x = x
        self.y = y
        self.color = color
        self.caption = caption
        self.oc = outline_color
        self.cc = check_color
        self.fs = font_size
        self.fc = font_color
        self.to = text_offset
        if not disable_pass:
            self.test_pass     = "PASS        "
            self.test_fail     = "FAIL        "
            self.test_untested = "WAITING"
        else:
            self.test_pass = ""
            self.test_fail = ""
            self.test_untested = ""
        # checkbox object
        self.checkbox_obj = pg.Rect(self.x, self.y, 35, 35)
        self.checkbox_outline = self.checkbox_obj.copy()
        self.write_box =  pg.Rect(self.x + 42, self.y, 120, 35)
        # variables to test the different states of the checkbox
        self.checked = check    
        if check:
            self.unchecked = False
            self.Unattended = False
        else:
            self.Unattended = True
            self.unchecked = True
        self.active = False
        self.click = False

    def _draw_button_text(self):
        self.font = pg.font.Font(None, self.fs)
        self.font_pos = (self.x + 45, self.y)
        if self.checked:
            pg.draw.rect(self.surface, (255,255,255), self.write_box)
            self.font_surf = self.font.render(self.test_pass, True, green)
            self.surface.blit(self.font_surf, self.font_pos)
        else:
            pg.draw.rect(self.surface, (255,255,255), self.write_box)
            self.font_surf = self.font.render(self.test_fail, True, red)
            self.surface.blit(self.font_surf, self.font_pos)
        if self.Unattended:
            pg.draw.rect(self.surface, (255,255,255), self.write_box)
            self.font_surf = self.font.render(self.test_untested, True, orange)
            self.surface.blit(self.font_surf, self.font_pos)
        
    def render_checkbox(self):
        if self.checked:
            pg.draw.rect(self.surface, self.color, self.checkbox_obj)
            pg.draw.rect(self.surface, self.oc, self.checkbox_outline, 3)
            pg.draw.circle(self.surface, self.cc, (self.x + 17, self.y + 17), 10)
        elif self.unchecked:
            pg.draw.rect(self.surface, self.color, self.checkbox_obj)
            pg.draw.rect(self.surface, self.oc, self.checkbox_outline, 3)
            self.checked = False
        self._draw_button_text()

    def _update(self, event_object):
        x, y = event_object.pos
        px, py, w, h = self.checkbox_obj  # getting check box dimensions
        if px < x < px + w and py < y < py + h:
            self.active = True
        else:
            self.active = False

    def _mouse_up(self):
            if self.active and not self.checked and self.click:
                self.checked = True
            elif self.active and self.checked and self.click:
                self.checked = False
                self.unchecked = True

    def update_checkbox(self, event_object):
        if event_object.type == pg.MOUSEBUTTONDOWN:
            self.click = True
        if event_object.type == pg.MOUSEBUTTONUP:
            self._mouse_up()
        if event_object.type == pg.MOUSEMOTION:
            self._update(event_object)

    def update_checkbox_rel(self, event_object, cb1, cb2):
        if event_object.type == pg.MOUSEBUTTONDOWN:
            self.click = True
        if event_object.type == pg.MOUSEBUTTONUP:
            self._mouse_up()
        if event_object.type == pg.MOUSEMOTION:
            self._update(event_object)
        if self.checked:
            cb1.uncheck()
            cb2.uncheck()

    def uncheck(self):
        self.Unattended = False
        self.checked = False
        self.unchecked = True
    
    def check(self):
        self.Unattended = False
        self.checked = True
        self.unchecked = False
    
    def setUnattended(self):
        self.Unattended = True
        self.checked = False
        self.unchecked = True
    
    def is_checked(self):
        return self.checked
