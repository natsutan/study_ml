import PIL
import os
from PIL import Image, ImageDraw

original_image = 'img/spinner_small.jpg'
LOOP = 1000

class Snake():
    def __init__(self, initial_value, im, k1=0.5, k2=0.1, k3=0.0):
        self.snake = initial_value
        self._i = 0
        self._last = False
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.elastic_dxdy = []
        self.external_dxdy = []
        self.im_gray = im.convert("L")

    def __iter__(self):
        return self

    def __next__(self):
        if self._last:
            self._last = False
            self._i = 0
            raise StopIteration()

        if self._i == len(self.snake) - 1:
            self._last = True
            return self.snake[self._i-1], self.snake[self._i], self.snake[0]

        val = self.snake[self._i-1], self.snake[self._i], self.snake[self._i+1]
        self._i += 1
        return val

    def update(self):
        self.calc_elastic_dxdy()
        self.calc_external_dxdy()
        self.update_by_dxdy()

    def calc_elastic_dxdy(self):
        self.elastic_dxdy = []
        for p0, p1, p2 in self:
            dx = self.k2 * 2 * self.k1 * ((p0[0] - p1[0]) + (p2[0] - p1[0]))
            dy = self.k2 * 2 * self.k1 * ((p0[1] - p1[1]) + (p2[1] - p1[1]))
            self.elastic_dxdy.append((dx, dy))


    def calc_external_dxdy(self):
        self.external_dxdy = []
        for x, y in self.snake:
            right = self.im_gray.getpixel((x+1, y))
            left = self.im_gray.getpixel((x-1, y))
            top = self.im_gray.getpixel((x, y-1))
            bottom = self.im_gray.getpixel((x, y+1))
            dx = 0.5 * self.k3 * (right - left)
            dy = 0.5 * self.k3 * (bottom - top)
            self.external_dxdy.append((dx, dy))

    def update_by_dxdy(self):
        for i in range(len(self.snake)):
            nx = min(self.snake[i][0] + self.elastic_dxdy[i][0] + self.external_dxdy[i][0], self.im_gray.width-2)
            ny = min(self.snake[i][1] + self.elastic_dxdy[i][1] + self.external_dxdy[i][1], self.im_gray.height-2)
            nx = max(1, nx)
            ny = max(1, ny)

            self.snake[i] = (nx, ny)



def plot_snake(im, snake, outfile):
    draw = ImageDraw.Draw(im)
    for _, p0, p1 in snake:
        draw.line((p0, p1), fill=(255, 0, 0))

    im.save(outfile)


def main():
    im = Image.open(original_image)
    initial_snake = [(1, 10), (10, 10), (20, 10), (30, 10),(40, 10),(50, 10),(60, 10),(70, 10),(80, 10),(90, 10), (100, 10),
                     (110, 10),(120, 10),(130, 10),(140, 10),(150, 10),(160, 10),(170, 10),(180, 10),(190, 10), (200, 10),
                     (210, 10), (220, 10), (230, 10), (240, 10), (250, 10), (260, 10), (270, 10), (280, 10), (290, 10),(300, 10),

                     (290, 15), (230, 40), (280, 100), (290, 150), (290, 200), (280, 250), (290, 300), (280, 360), (300, 400),
                      (250, 390), (200, 380), (150, 390), (50, 370), (20, 350),
                      (10, 300), (15, 260), (20, 210), (10, 150), (30, 100), (20, 50)]
    snake = Snake(initial_snake, im)
    #plot_snake(im, snake, os.path.join('out', 'snake.png'))

    for i in range(LOOP):
        snake.update()
        if i % 100 == 0:
            plot_snake(im, snake, os.path.join('out', 'snake{}.png'.format(i)))


if __name__ == '__main__':
    main()

