import logging
import time
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

import networks
from utils import (
    preprocess_image_train,
    preprocess_image_test,
    generator_loss,
    calc_cycle_loss,
    discriminator_loss,
    identity_loss,
)

logging.basicConfig(level=logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
NUM_SAMPLES = 3


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="models/cycle")
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
    parser.add_argument("--lr-generator", default=2e-4, type=float)
    parser.add_argument("--lr-discriminator", default=2e-4, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    dataset, metadata = tfds.load(
        "cycle_gan/{}".format(args.dataset), with_info=True, as_supervised=True
    )

    train_x, train_y = dataset["trainA"], dataset["trainB"]
    test_x, test_y = dataset["testA"], dataset["testB"]

    train_x = (
        train_x.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(args.batch_size)
    )
    train_y = (
        train_y.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(args.batch_size)
    )
    test_x = test_x.map(preprocess_image_test).batch(NUM_SAMPLES)
    test_y = test_y.map(preprocess_image_test).batch(NUM_SAMPLES)

    ds = tf.data.Dataset.zip((train_x, train_y))

    sample_x = next(iter(test_x))
    sample_y = next(iter(test_y))

    generator_g = networks.GeneratorNet()
    generator_f = networks.GeneratorNet()

    discriminator_x = networks.DiscriminatorNet()
    discriminator_y = networks.DiscriminatorNet()

    generator_g_optimizer = tf.keras.optimizers.Adam(
        args.lr_generator, beta_1=0.5
    )
    generator_f_optimizer = tf.keras.optimizers.Adam(
        args.lr_generator, beta_1=0.5
    )

    discriminator_x_optimizer = tf.keras.optimizers.Adam(
        args.lr_discriminator, beta_1=0.5
    )
    discriminator_y_optimizer = tf.keras.optimizers.Adam(
        args.lr_discriminator, beta_1=0.5
    )

    ckpt = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer,
    )

    manager = tf.train.CheckpointManager(ckpt, args.log_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    generator_g_loss = tf.keras.metrics.Mean(name="generator_g_loss")
    generator_f_loss = tf.keras.metrics.Mean(name="generator_f_loss")
    discriminator_x_loss = tf.keras.metrics.Mean(name="discriminator_x_loss")
    discriminator_y_loss = tf.keras.metrics.Mean(name="discriminator_y_loss")

    summary_writer = tf.summary.create_file_writer(args.log_dir)

    with summary_writer.as_default():
        tf.summary.image("Sample X", sample_x * 0.5 + 0.5, step=0)
        tf.summary.image("Sample Y", sample_y * 0.5 + 0.5, step=0)

    @tf.function
    def train_step(real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(
                real_x, cycled_x
            ) + calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = (
                gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            )
            total_gen_f_loss = (
                gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
            )

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(
            total_gen_g_loss, generator_g.trainable_variables
        )
        generator_f_gradients = tape.gradient(
            total_gen_f_loss, generator_f.trainable_variables
        )

        discriminator_x_gradients = tape.gradient(
            disc_x_loss, discriminator_x.trainable_variables
        )
        discriminator_y_gradients = tape.gradient(
            disc_y_loss, discriminator_y.trainable_variables
        )

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients, generator_g.trainable_variables)
        )

        generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients, generator_f.trainable_variables)
        )

        discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients, discriminator_x.trainable_variables)
        )

        discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients, discriminator_y.trainable_variables)
        )

        # Log the losses
        generator_g_loss(total_gen_g_loss)
        generator_f_loss(total_gen_f_loss)
        discriminator_x_loss(disc_x_loss)
        discriminator_y_loss(disc_y_loss)

    lr_generator_delta = args.lr_generator / (args.epochs / 2)
    lr_discriminator_delta = args.lr_discriminator / (args.epochs / 2)

    for epoch in range(args.epochs):
        start_time = time.time()

        for image_x, image_y in ds:
            train_step(image_x, image_y)

        if (epoch + 1) % 5 == 0:
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        # Decay the learning rate
        if epoch > args.epochs / 2:
            generator_g_optimizer.learning_rate.assign_sub(lr_generator_delta)
            generator_f_optimizer.learning_rate.assign_sub(lr_generator_delta)

            discriminator_x_optimizer.learning_rate.assign_sub(
                lr_discriminator_delta
            )
            discriminator_y_optimizer.learning_rate.assign_sub(
                lr_discriminator_delta
            )

        with summary_writer.as_default():
            tf.summary.scalar(
                "generator/g_loss", generator_g_loss.result(), step=epoch
            )
            tf.summary.scalar(
                "generator/f_loss", generator_f_loss.result(), step=epoch
            )
            tf.summary.scalar(
                "discriminator/x_loss",
                discriminator_x_loss.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator/y_loss",
                discriminator_y_loss.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "learning_rate/generator",
                generator_g_optimizer.learning_rate.numpy(),
                step=epoch,
            )
            tf.summary.scalar(
                "learning_rate/discriminator",
                discriminator_x_optimizer.learning_rate.numpy(),
                step=epoch,
            )
            tf.summary.image(
                "Generated Y", generator_g(sample_x) * 0.5 + 0.5, step=epoch
            )
            tf.summary.image(
                "Generated X", generator_f(sample_y) * 0.5 + 0.5, step=epoch
            )

        template = "Epoch {} in {:.0f} sec, Gen G Loss: {}, Gen F Loss: {}, Disc X Loss: {}, Disc Y Loss: {}"
        print(
            template.format(
                (epoch + 1),
                time.time() - start_time,
                generator_g_loss.result(),
                generator_f_loss.result(),
                discriminator_x_loss.result(),
                discriminator_y_loss.result(),
            )
        )

        generator_g_loss.reset_states(),
        generator_f_loss.reset_states(),
        discriminator_x_loss.reset_states(),
        discriminator_y_loss.reset_states(),
