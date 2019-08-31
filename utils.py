import tensorflow as tf
from PIL import Image

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 10


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


def random_crop(image, size):
    cropped_image = tf.image.random_crop(image, size=size)

    return cropped_image


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    image = tf.image.resize(
        image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    image = random_crop(image, size=[256, 256, 3])
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image, _):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, _):
    image = normalize(image)
    return image


def save_img(image, path):
    image = tf.cast((tf.squeeze(image) + 1) * 127.5, tf.uint8).numpy()
    img = Image.fromarray(image, mode="RGB")
    img.save(path)
