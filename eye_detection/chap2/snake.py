import PIL
import os
from PIL import Image, ImageDraw

original_image = 'img/spinner_small.jpg'

class Snake():
    def __init__(self, inisital_value):
        self.snake = inisital_value
        self._i = 0
        self._last = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._last:
            self._last = False
            self._i = 0
            raise StopIteration()

        if self._i == len(self.snake) - 1:
            self._last = True
            return self.snake[self._i], self.snake[0]

        val = self.snake[self._i], self.snake[self._i+1]
        self._i += 1
        return val


def plot_snake(im, snake, outfile):
    draw = ImageDraw.Draw(im)
    for p0, p1 in snake:
        draw.line((p0, p1), fill=(255, 0, 0))

    im.save(outfile)


def main():
    im = Image.open(original_image)
    initial_snake = [(1,10), (50, 10), (75, 10), (100, 10), (150, 10),
                      (290, 5), (230, 40), (280, 100), (290, 150), (290, 200), (280, 250), (290, 300), (280, 360), (300, 400),
                      (250, 390), (200, 380), (150, 390), (50, 370), (20, 350),
                      (10, 300), (15, 260), (20, 210), (10, 150), (30, 100), (20, 50) ]
    snake = Snake(initial_snake)

    plot_snake(im, snake, os.path.join('out', 'snake.png'))








if __name__ == '__main__':
    main()

