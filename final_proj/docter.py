import math
import sys

#import .base
from final_proj.base.pygamewrapper import PyGameWrapper
import numpy as np
import pygame
from pygame.constants import K_w, K_s
from final_proj.utils.vec2d import vec2d
import random





class DocterPlayer(pygame.sprite.Sprite):

    def __init__(self, speed, SCREEN_WIDTH, SCREEN_HEIGHT):
        #the normal blood pressure is from 80 to 120
        self.y_loc = np.hstack((np.hstack((np.sort(np.random.uniform(SCREEN_HEIGHT-120, SCREEN_HEIGHT-80, 25))[::-1], np.sort(np.random.uniform(SCREEN_HEIGHT-119, SCREEN_HEIGHT-90, 25)))), np.sort(np.random.uniform(SCREEN_HEIGHT-89, SCREEN_HEIGHT-80, 20))[::-1]))
        #blood pressure for hypotension is from 50 to 90
        

        pygame.sprite.Sprite.__init__(self)
        self.trajectory = []
        pos_init = (int(SCREEN_WIDTH * 0.35), SCREEN_HEIGHT / 2)
        self.pos = vec2d(pos_init)
        self.speed = speed
        self.climb_speed = speed * -0.875  # -0.0175
        self.fall_speed = speed * 0.09  # 0.0019
        self.momentum = 0

        self.width = SCREEN_WIDTH * 0.05
        self.height = SCREEN_HEIGHT * 0.05

        image = pygame.Surface((self.width, self.height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, self.width, self.height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, is_climbing, dt, y_pos, hypo_sig):
        #self.momentum += (self.climb_speed if is_climbing else self.fall_speed) * dt
        #self.momentum *= 0.99
        if(hypo_sig):
            self.y_loc = self.y_loc+15
        self.pos.y = self.y_loc[y_pos]
        #self.pos.y += self.momentum
        if(is_climbing):
            self.y_loc = self.y_loc-10
        self.rect.center = (self.pos.x, self.pos.y)
        self.trajectory.append([self.pos.x, self.pos.y])
        
        self.trajectory = [(x[0] - self.speed*10, x[1]) for x in self.trajectory]
        if len(self.trajectory) > 100:  # Store last 100 positions
            self.trajectory.pop(0)



class BPWaveform(PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    """

    def __init__(self, width=48, height=48):
        actions = {
            "heal": K_w
        }
        self.iter = 0
        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.y_pos = 0
        self.hypo_sig = False

        self.is_climbing = False
        self.speed = 0.0004 * width

    def _handle_player_events(self):
        self.is_climbing = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['heal']:
                    self.is_climbing = True

    def getGameState(self):
        """
        Returns
        -------

        dict
            * player y position.


        """
        state = {
            "player_y": self.player.pos.y,

        }

        return state


    def getActions(self):
        return self.actions.values()

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives <= 0.0 or self.score>=1000

    def init(self):
        self.score = 0.0
        self.lives = 1.0

        self.player = DocterPlayer(
            self.speed,
            self.width,
            self.height
        )

        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)


    def reset(self):
        self.init()

    def step(self, dt):

        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        self.player.update(self.is_climbing, dt, self.y_pos,self.hypo_sig)

        self.y_pos+=1
        if(self.hypo_sig == True):
            self.hypo_sig = False
        else:
            #a chance of hypotension
            random_number = random.randint(1, 500)
            if(random_number == 50):
                self.hypo_sig = True
        
        if(self.y_pos == 70):
            self.iter+=1
            self.y_pos = 0
            self.score += self.rewards["positive"]
        



        if self.player.pos.y > self.height-65:  # its below the lowest possible block
            self.score += self.rewards["negative"]
            self.lives -=1
        elif self.player.pos.y < self.height-150:
            self.score += self.rewards["negative"]
            self.lives -=1
        
        if len(self.player.trajectory) > 1:  # Make sure there are at least two points
            pygame.draw.lines(self.screen, (255, 0, 0), False, self.player.trajectory, 2)
        line_color = (0, 0, 255)  # RGB for blue

        # Function to draw a line and its label
        def draw_line_and_label(screen, y, color, label_text):
            font = pygame.font.SysFont(None, 24)  # You can choose another font and size
            label = font.render(label_text, True, (0, 255, 0))  # Render the text

            # Draw the line
            start_point = (0, y)
            end_point = (screen.get_width(), y)
            pygame.draw.line(screen, color, start_point, end_point, 2)

            # Blit the label onto the screen near the line
            screen.blit(label, (10, y - 10))  # Adjust the position as needed

        # Draw two lines with labels
        draw_line_and_label(self.screen, self.height-65, line_color, "Blood pressure 65, hypotension")
        draw_line_and_label(self.screen, self.height-150, line_color, "Blood pressure 150")

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = BPWaveform(width=500, height=500)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()
    def transform_y(y):
        return 500 - y
    while True:
        if game.game_over():
            game.reset()
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
