import os

import tensorflow as tf
import tensorflow_datasets as tfds

from argparse import ArgumentParser

import networks
from utils import preprocess_image_test, save_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--output-dir", default="images")
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument(
        "--dataset",
        default="horse2zebra",
        choices=[
            "apple2orange",
            "summer2winter_yosemite",
            "horse2zebra",
            "monet2photo",
            "cezanne2photo",
            "ukiyoe2photo",
            "vangogh2photo",
        ],
    )
    args = parser.parse_args()

    generator_g = networks.GeneratorNet()
    generator_f = networks.GeneratorNet()

    ckpt = tf.train.Checkpoint(
        generator_g=generator_g, generator_f=generator_f
    )
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    dataset, metadata = tfds.load(
        "cycle_gan/{}".format(args.dataset), with_info=True, as_supervised=True
    )

    test_x, test_y = dataset["testA"], dataset["testB"]
    test_x = test_x.map(preprocess_image_test).batch(1)
    test_y = test_y.map(preprocess_image_test).batch(1)
    ds = tf.data.Dataset.zip((test_x, test_y))

    for i, (image_x, image_y) in enumerate(ds.take(args.num_samples)):
        fake_y = generator_g(image_x)
        cycled_x = generator_f(fake_y)

        save_img(image_x, os.path.join(args.output_dir, "real_x_{}.png".format(i)))
        save_img(fake_y, os.path.join(args.output_dir, "fake_y_{}.png".format(i)))
        save_img(cycled_x, os.path.join(args.output_dir, "cycled_x_{}.png".format(i)))

        fake_x = generator_f(image_y)
        cycled_y = generator_g(fake_x)

        save_img(image_y, os.path.join(args.output_dir, "real_y_{}.png".format(i)))
        save_img(fake_x, os.path.join(args.output_dir, "fake_x_{}.png".format(i)))
        save_img(cycled_y, os.path.join(args.output_dir, "cycled_y_{}.png".format(i)))
