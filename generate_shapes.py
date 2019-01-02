#!/usr/bin/env python3

import argparse
import collections
import json
import os
import time

import numpy as np
import scipy.misc
import skimage.draw

NUMBER_OF_ATTEMPS_TO_FIT_SHAPES = 100

Label = collections.namedtuple('Label', 'category, x1, x2, y1, y2')


def _generate_rectangle_mask(x_0,
                             y_0,
                             image_width,
                             image_height,
                             image_depth,
                             color,
                             min_dimension,
                             max_dimension):
    # (x_0, y_0) is the top left corner.
    available_width = min(image_width - x_0, max_dimension)
    if available_width < min_dimension:
        raise ArithmeticError
    available_height = min(image_height - y_0, max_dimension)
    if available_height < min_dimension:
        raise ArithmeticError
    w = np.random.randint(min_dimension, available_width + 1)
    h = np.random.randint(min_dimension, available_height + 1)
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    mask[y_0:y_0 + h, x_0:x_0 + w] = color
    assert mask.sum() > 0
    label = Label('rectangle', x_0, x_0 + w, y_0, y_0 + h)

    return mask, label


def _generate_circle_mask(x_0,
                          y_0,
                          image_width,
                          image_height,
                          image_depth,
                          color,
                          min_dimension,
                          max_dimension):
    if min_dimension == 1 or max_dimension == 1:
        raise ValueError('dimension must be > 1 for circles')
    min_dimension /= 2
    max_dimension /= 2
    # (x_0, y_0) is the center
    left = x_0
    right = image_width - x_0
    top = y_0
    bottom = image_height - y_0
    available_radius = min(left, right, top, bottom, max_dimension)
    if available_radius < min_dimension:
        raise ArithmeticError
    radius = np.random.randint(min_dimension, available_radius + 1)
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    circle = skimage.draw.circle(y_0, x_0, radius)
    mask[circle] = color
    assert mask.sum() > 0
    label = Label('circle', x_0 - radius + 1, x_0 + radius, y_0 - radius + 1,
                  y_0 + radius)

    return mask, label


def _generate_triangle_mask(x_0,
                            y_0,
                            image_width,
                            image_height,
                            image_depth,
                            color,
                            min_dimension,
                            max_dimension):
    if min_dimension == 1 or max_dimension == 1:
        raise ValueError('dimension must be > 1 for circles')
    # (x_0, y_0) is the bottom left corner.
    # We're making an equilateral triangle.
    available_side = min(image_width - x_0, y_0 + 1, max_dimension)
    if available_side < min_dimension:
        raise ArithmeticError
    side = np.random.randint(min_dimension, available_side + 1)
    triangle_height = int(np.ceil(np.sqrt(3 / 4) * side))
    mask = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    triangle = skimage.draw.polygon([y_0, y_0 - triangle_height, y_0],
                                    [x_0, x_0 + side // 2, x_0 + side])
    mask[triangle] = color
    assert mask.sum() > 0
    label = Label('triangle', x_0, x_0 + side, y_0 - triangle_height, y_0)

    return mask, label


SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


def generate_random_color(gray, min_intensity):
    size = 1 if gray else 3
    return np.random.randint(min_intensity, 255, size=size)


def _generate_image(width,
                    height,
                    number_of_shapes,
                    min_dimension,
                    max_dimension,
                    gray,
                    shape,
                    min_intensity,
                    allow_overlap):
    depth = 1 if gray else 3
    image = np.ones((height, width, depth), dtype=np.uint8) * 255
    labels = []
    for _ in range(number_of_shapes):
        # Pick start coordinates.
        x = np.random.randint(width)
        y = np.random.randint(width)
        color = generate_random_color(gray, min_intensity)
        if shape is None:
            shape_generator = np.random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[shape]
        try:
            mask, label = shape_generator(x,
                                          y,
                                          width,
                                          height,
                                          depth,
                                          color,
                                          min_dimension,
                                          max_dimension)
        except ArithmeticError:
            # Couldn't fit the shape, skip it.
            pass
        else:
            assert mask.sum() > 0, mask
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or image[mask.nonzero()].min() == 255:
                image -= mask  # This inverts the color (it's random anyway).
                labels.append(label)

    return image, labels


def verify_arguments(width, height, min_dimension, min_intensity):
    if min_dimension > width or min_dimension > height:
        raise ValueError(
            'Minimum dimension must be less than width and height')
    if not (0 <= min_intensity <= 255):
        raise ValueError('Minimum intensity must be in interval [0, 255]')


def generate_shapes(number_of_images,
                    width,
                    height,
                    max_shapes,
                    min_shapes=1,
                    min_dimension=2,
                    max_dimension=None,
                    gray=False,
                    shape=None,
                    min_intensity=32,
                    allow_overlap=False):
    if max_dimension is None:
        max_dimension = max(height, width)
    if min_shapes > max_shapes:
        max_shapes = min_shapes
    verify_arguments(width, height, min_dimension, min_intensity)

    images = []
    labels = []
    for _ in range(number_of_images):
        for _ in range(NUMBER_OF_ATTEMPS_TO_FIT_SHAPES):
            number_of_shapes = np.random.randint(min_shapes, max_shapes + 1)
            image, image_labels = _generate_image(width,
                                                  height,
                                                  number_of_shapes, min_dimension,
                                                  max_dimension,
                                                  gray,
                                                  shape,
                                                  min_intensity,
                                                  allow_overlap)
            if image_labels:
                images.append(image)
                labels.append(image_labels)
                break
    return images, labels


def save_images_and_labels(output_directory, images, labels):
    if os.path.exists(output_directory):
        assert os.path.isdir(output_directory)
    else:
        os.makedirs(output_directory)
    print('Saving to {0} ...'.format(os.path.abspath(output_directory)))
    for number, image in enumerate(images):
        path = os.path.join(output_directory, '{0}.png'.format(number))
        scipy.misc.imsave(path, image)
    labels_file_path = os.path.join(output_directory, 'labels.json')
    with open(labels_file_path, 'w') as labels_file:
        new_labels = []
        for image_labels in labels:
            new_image_labels = []
            for shape_label in image_labels:
                new_shape_labels = shape_label._asdict()
                new_shape_labels['class'] = new_shape_labels.pop('category')
                new_image_labels.append(new_shape_labels)
            new_labels.append(dict(boxes=new_image_labels))
        json.dump(new_labels, labels_file, indent=4)


def show_images(images):
    for image in images:
        scipy.misc.toimage(image).show()


def parse():
    parser = argparse.ArgumentParser(
        description='Generate Toy Object Detection Dataset')
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        required=True,
        help='The number of images to generate')
    parser.add_argument(
        '--width',
        type=int,
        default=128,
        help='The width of generated images (128)')
    parser.add_argument(
        '--height',
        type=int,
        default=128,
        help='The height of generated images (128)')
    parser.add_argument(
        '--max-shapes',
        type=int,
        default=10,
        help='The maximum number of shapes per image (10)')
    parser.add_argument(
        '--min-shapes',
        type=int,
        default=1,
        help='The maximum number of shapes per image (1)')
    parser.add_argument(
        '--min-dimension',
        type=int,
        default=10,
        help='The minimum dimension of a shape (10)')
    parser.add_argument(
        '--max-dimension',
        type=int,
        help='The maximum dimension of a shape (None)')
    parser.add_argument(
        '--min-intensity',
        type=int,
        default=128,
        help='The minimum intensity (0-255) for a pixel channel (128)')
    parser.add_argument(
        '--gray', action='store_true', help='Make all shapes grayscale')
    parser.add_argument(
        '--shape',
        choices=SHAPE_GENERATORS.keys(),
        help='Generate only this kind of shape')
    parser.add_argument(
        '-o', '--output-dir', help='The output directory where to save images')
    parser.add_argument(
        '--allow-overlap',
        action='store_true',
        help='Allow shapes to overlap on images')

    return parser.parse_args()


def main():
    options = parse()
    start = time.time()
    images, labels = generate_shapes(number_of_images=options.number,
                                     width=options.width,
                                     height=options.height,
                                     min_shapes=options.min_shapes,
                                     max_shapes=options.max_shapes,
                                     min_dimension=options.min_dimension,
                                     max_dimension=options.max_dimension,
                                     gray=options.gray,
                                     shape=options.shape,
                                     min_intensity=options.min_intensity,
                                     allow_overlap=options.allow_overlap)
    end = time.time() - start
    print('Generated {0} images in {1:.2f}s'.format(len(images), end))
    if options.output_dir is None:
        show_images(images)
    else:
        save_images_and_labels(options.output_dir, images, labels)


if __name__ == '__main__':
    main()
