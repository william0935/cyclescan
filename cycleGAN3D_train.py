import SimpleITK as sitk
import nibabel as nib
import numpy as np 
import os
import random
import glob
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from tqdm import tqdm
import itertools
import pydicom as pdm
import tensorflow as tf
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, concatenate, Conv3DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
tf.random.set_seed(42)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### LOAD DATA ### 
'''
Preprocessing steps
1. Get image (convert image from .nii file to tensor)
2. crop image (256,256,32) (h,w,d)
3. Add random jitter and flip image
4. normalize image to [-1,1] pixel values
'''
def get_image(nii_path):
    img_nii = nib.load(nii_path).get_fdata()
    return tf.convert_to_tensor(img_nii,dtype=np.float32)

def crop_image(img, crop_range = [256, 256, 32]):
    img_dims = img.shape
    #Slices
    slice_start,slice_stop = img_dims[2]//2 - crop_range[2]//2, img_dims[2]//2 + crop_range[2]//2

    # Row
    row_start,row_stop = img_dims[0]//2 - crop_range[0]//2, img_dims[0]//2 + crop_range[0]//2

    # Columns 
    col_start,col_stop = img_dims[1]//2 - crop_range[1]//2, img_dims[1]//2 + crop_range[1]//2
    
    img_cropped = img[row_start:row_stop, col_start:col_stop,slice_start:slice_stop]

    return img_cropped

def random_jitter(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.reverse(image, axis=[1]) 
    return image

def normalize_image(image):

    image = (image / 127.5) - 1.0 # Normalize to [-1,1]
    image = image[..., np.newaxis] # Add channel Axis
    
    return image

def preprocess_image_train(image):
    image = crop_image(image)
    image = random_jitter(image)
    image = normalize_image(image)
    
    return image    

def preprocess_image_test(image):
    image = crop_image(image)
    image = normalize_image(image)
    
    return image    

def load_nii_and_preprocess(folder_str, shuffle = True, train=True):
    img_data = [] # initialize a list for tensors 

    
    file_list = os.listdir(folder_str) 
    
    if shuffle:
        random.shuffle(file_list)
    
    for _, file in enumerate(file_list):
        if file.endswith(".nii"):
            img = get_image(os.path.join(folder_str,file))
            if train:
                img = preprocess_image_train(img)
            else:
                img = preprocess_image_test(img)
            img = tf.transpose(img, perm=[2, 0, 1, 3])
            img_data.append(img)

    all_data = tf.stack(img_data)
    tf_data = tf.data.Dataset.from_tensor_slices(all_data) # (D, H, W, C)
    return tf_data
    
    
### Build Model ###
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

### Downsample Block ###
def downsample(filters, size, norm_type='instancenorm', apply_norm=True):
    """Downsamples an input.

    Conv3D => Batchnorm => LeakyRelu

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(filters, size, strides=(2,2,2), padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

### Upsample Block ###
def upsample(filters, size, norm_type='instancenorm', apply_dropout=False):
    """Upsamples an input.

    Conv3DTranspose => Batchnorm => Dropout => Relu

    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

    Returns:
    Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3DTranspose(filters, size, strides=(2,2,2),
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

### U-Net Generator ###
def unet_generator(output_channels, norm_type='batchnorm'):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

        Args:
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

        Returns:
        Generator model
        """

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
        downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
        downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
        downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 512)
        upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 16, 16, 512)
        upsample(256, 4, norm_type),  # (bs, 32, 32, 256)
        upsample(128, 4, norm_type),  # (bs, 64, 64, 128)
        upsample(64, 4, norm_type),  # (bs, 128, 128, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 1)

    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
    x = inputs
    print(x.shape)
    # Downsampling through the model

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    print(x.shape)
    return tf.keras.Model(inputs=inputs, outputs=x)

### Discriminator ###
def discriminator(norm_type='batchnorm', target=True):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

    Returns:
    Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, None, 1], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv3D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)

# Apply Generator and Discriminator 
OUTPUT_CHANNELS = 1  # For black-and-white 3D images

generator_g = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = discriminator(norm_type='instancenorm', target=False)
discriminator_y = discriminator(norm_type='instancenorm', target=False) 

### LOSS FUNCTIONS ###
LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits = True)

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
    loss= tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

def kl_divergence_histogram(real_img, generated_img, num_bins=256):
    """
    Computes KL divergence between voxel intensity histograms of two 3D volumes.
    Inputs are assumed to be in [-1, 1], and will be rescaled to [0, 1].
    """
    # Flatten and rescale from [-1, 1] -> [0, 1]
    real_flat = tf.reshape((real_img + 1.0) / 2.0, [-1])
    gen_flat = tf.reshape((generated_img + 1.0) / 2.0, [-1])

    # Compute histograms
    hist_real = tf.histogram_fixed_width(real_flat, [0.0, 1.0], nbins=num_bins)
    hist_gen = tf.histogram_fixed_width(gen_flat, [0.0, 1.0], nbins=num_bins)

    # Normalize to get probability distributions
    p = tf.cast(hist_real, tf.float32) / tf.reduce_sum(hist_real)
    q = tf.cast(hist_gen, tf.float32) / tf.reduce_sum(hist_gen)

    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    kl_div = tf.reduce_sum(p * tf.math.log((p + epsilon) / (q + epsilon)))

    return kl_div

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                        generator_f=generator_f,
                        discriminator_x=discriminator_x,
                        discriminator_y=discriminator_y,
                        generator_g_optimizer=generator_g_optimizer,
                        generator_f_optimizer=generator_f_optimizer,
                        discriminator_x_optimizer=discriminator_x_optimizer,
                        discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
else:
    print('No checkpoint found, starting training from scratch.')

EPOCHS = 200
def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12,12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.title(title[i])
        plt.imshow(display_list[i]*0.5+0.5, cmap='gray') # getting pixel values between [0,1] to plot it 
        plt.axis('off')
    plt.show()
    

#### TRAIINING FUNCTION ####
@tf.function

def train_step(real_x, real_y):
    with tf.GradientTape(persistent = True) as tape: #persistent is set to True because the tape is used more than once to calculate the gradients 

        # Generator G translates A -> B (X -> Y)
        # Generator F translates B -> A ( Y -> X)

        fake_y = generator_g(real_x, training = True)
        cycled_x = generator_f(fake_y, training = True)

        fake_x = generator_f(real_y, training = True)
        cycled_y = generator_g(fake_x, training = True)

        # same_x and same_y are used for identity loss 
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training = True)

        disc_real_x = discriminator_x(real_x, training = True)
        disc_real_y = discriminator_y(real_y, training = True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training = True)

        #calculate loss 
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        kl_hist_x = kl_divergence_histogram(real_x, fake_x)
        kl_hist_y = kl_divergence_histogram(real_y, fake_y)


    # Calculate the gradients for generator and discriminators 
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))
    
    return total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss, kl_hist_x, kl_hist_y
### VALIDATION FUNCTION ### 
@tf.function
def validation_step(real_x, real_y):
    # Generator G translates A -> B (X -> Y)
    # Generator F translates B -> A ( Y -> X)

    fake_y = generator_g(real_x, training = False)
    cycled_x = generator_f(fake_y, training = False)

    fake_x = generator_f(real_y, training = False)
    cycled_y = generator_g(fake_x, training = False)

    # same_x and same_y are used for identity loss 
    same_x = generator_f(real_x, training=False)
    same_y = generator_g(real_y, training = False)

    disc_real_x = discriminator_x(real_x, training = False)
    disc_real_y = discriminator_y(real_y, training = False)

    disc_fake_x = discriminator_x(fake_x, training=False)
    disc_fake_y = discriminator_y(fake_y, training = True)

    #calculate loss 
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    kl_hist_x = kl_divergence_histogram(real_x, fake_x)
    kl_hist_y = kl_divergence_histogram(real_y, fake_y)

    return total_gen_g_loss, total_gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss, kl_hist_x, kl_hist_y

#### LOAD DATA #### 
base_path = 'data'

# Load your datasets
ciss_train = load_nii_and_preprocess(os.path.join(base_path, 'split_CISS/train'), shuffle=True, train=True)
ciss_val = load_nii_and_preprocess(os.path.join(base_path, 'split_CISS/val'), shuffle=False, train=False)

dess_train = load_nii_and_preprocess(os.path.join(base_path, 'split_DESS/train'), shuffle=True, train=True)
dess_val = load_nii_and_preprocess(os.path.join(base_path, 'split_DESS/val'), shuffle=False, train=False)
# dess_test = load_nii_and_preprocess(os.path.join(base_path, 'split_DESS/test'), shuffle=False, train=False)

c_train = ciss_train.batch(1)
d_train = dess_train.batch(1)
sample_ciss = next(iter(c_train))
sample_dess = next(iter(d_train))

print(sample_dess.shape)
history = {
    "gen_g_train": [],
    "gen_f_train": [],
    "cycle_train": [],
    "disc_x_train": [],
    "disc_y_train": [],
    "kl_ciss_train": [],
    "kl_dess_train": [],
    
    "gen_g_val": [],
    "gen_f_val": [],
    "cycle_val": [],
    "disc_x_val": [],
    "disc_y_val": [],
    "kl_ciss_val": [],
    "kl_dess_val": []
}


BATCH_SIZE = 1
for epoch in range(EPOCHS):
    train_losses = {"gen_g": 0.0, "gen_f": 0.0, "cycle": 0.0, "disc_x": 0.0, "disc_y": 0.0, "kl_ciss": 0.0, "kl_dess":0.0}
    val_losses = {"gen_g": 0.0, "gen_f": 0.0, "cycle": 0.0, "disc_x": 0.0, "disc_y": 0.0, "kl_ciss": 0.0, "kl_dess":0.0}
    n_train = n_val = 0

    # Training loop
    for real_ciss_train, real_dess_train in tf.data.Dataset.zip((ciss_train.batch(BATCH_SIZE), dess_train.batch(BATCH_SIZE))): 
        gen_g_loss, gen_f_loss, cycle_loss, disc_x_loss, disc_y_loss, train_kl_ciss, train_kl_dess = train_step(real_ciss_train, real_dess_train)
        train_losses["gen_g"] += gen_g_loss.numpy()
        train_losses["gen_f"] += gen_f_loss.numpy()
        train_losses["cycle"] += cycle_loss.numpy()
        train_losses["disc_x"] += disc_x_loss.numpy()
        train_losses["disc_y"] += disc_y_loss.numpy()
        train_losses["kl_ciss"] += train_kl_ciss.numpy()
        train_losses["kl_dess"] += train_kl_dess.numpy()
        n_train += 1

    # Validation loop
    for real_ciss_val, real_dess_val in tf.data.Dataset.zip((ciss_val.batch(BATCH_SIZE), dess_val.batch(BATCH_SIZE))): 
        val_gen_g_loss, val_gen_f_loss, val_cycle_loss, val_disc_x_loss, val_disc_y_loss,val_kl_ciss,val_kl_dess = validation_step(real_ciss_val, real_dess_val)
        val_losses["gen_g"] += val_gen_g_loss.numpy()
        val_losses["gen_f"] += val_gen_f_loss.numpy()
        val_losses["cycle"] += val_cycle_loss.numpy()
        val_losses["disc_x"] += val_disc_x_loss.numpy()
        val_losses["disc_y"] += val_disc_y_loss.numpy()
        val_losses["kl_ciss"] += val_kl_ciss.numpy()
        val_losses["kl_dess"] += val_kl_dess.numpy()
        n_val += 1

    # Store average losses
    for key in train_losses:
        history[f"{key}_train"].append(train_losses[key] / n_train)
        history[f"{key}_val"].append(val_losses[key] / n_val)

    # Print summary
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train - gen_g: {history['gen_g_train'][-1]:.4f}, gen_f: {history['gen_f_train'][-1]:.4f}, cycle: {history['cycle_train'][-1]:.4f}, disc_x: {history['disc_x_train'][-1]:.4f}, disc_y: {history['disc_y_train'][-1]:.4f}")
    print(f"  Val   - gen_g: {history['gen_g_val'][-1]:.4f}, gen_f: {history['gen_f_val'][-1]:.4f}, cycle: {history['cycle_val'][-1]:.4f}, disc_x: {history['disc_x_val'][-1]:.4f}, disc_y: {history['disc_y_val'][-1]:.4f}")

    # Generate images and save checkpoints periodically
    if (epoch + 1) % 10 == 0:
        generate_images(generator_g, sample_ciss.batch(BATCH_SIZE))
        generate_images(generator_f, sample_dess.batch(BATCH_SIZE))
        ckpt_manager.save()

# PLOTS 
def plot_loss(history, loss_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history[f"{loss_name}_train"], label=f"{loss_name} train")
    plt.plot(history[f"{loss_name}_val"], label=f"{loss_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{loss_name} Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot as a PNG file
    plot_filename = f"LossPlots/{loss_name}Loss.png"
    plt.savefig(plot_filename)

plot_loss(history, "gen_g")
plot_loss(history, "gen_f")
plot_loss(history, "cycle")
plot_loss(history, "disc_x")
plot_loss(history, "disc_y")
plot_loss(history, "kl_ciss")
plot_loss(history, "kl_dess")