import os
import sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
cur_path = os.getcwd()
DATA_DIR = os.path.join(cur_path, 'seq')

BATCH_SIZE = 32  # Batch size
ITERS = 200000  # How many iterations to train for
SEQ_LEN = 50  # Sequence length in characters
DIM = 512  # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 5  # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 14098  # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

lib.print_model_settings(locals().copy())

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random.normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

def Discriminator(inputs):
    output = tf.transpose(inputs, [0, 2, 1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output

# Build model using TF 2.x style
@tf.function
def train_discriminator(real_inputs_discrete):
    with tf.GradientTape() as tape:
        real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
        fake_inputs = Generator(BATCH_SIZE)
        
        disc_real = Discriminator(real_inputs)
        disc_fake = Discriminator(fake_inputs)
        
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        
        # WGAN lipschitz-penalty
        alpha = tf.random.uniform(
            shape=[BATCH_SIZE, 1, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_inputs - real_inputs
        interpolates = real_inputs + (alpha * differences)
        
        with tf.GradientTape() as tape2:
            tape2.watch(interpolates)
            disc_interpolates = Discriminator(interpolates)
        
        gradients = tape2.gradient(disc_interpolates, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty
    
    disc_params = lib.params_with_name('Discriminator')
    disc_grads = tape.gradient(disc_cost, disc_params)
    disc_optimizer.apply_gradients(zip(disc_grads, disc_params))
    
    return disc_cost

@tf.function
def train_generator():
    with tf.GradientTape() as tape:
        fake_inputs = Generator(BATCH_SIZE)
        disc_fake = Discriminator(fake_inputs)
        gen_cost = -tf.reduce_mean(disc_fake)
    
    gen_params = lib.params_with_name('Generator')
    gen_grads = tape.gradient(gen_cost, gen_params)
    gen_optimizer.apply_gradients(zip(gen_grads, gen_params))
    
    return gen_cost

# Optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

# Checkpoint
checkpoint = tf.train.Checkpoint(
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, './checkpoints', max_to_keep=4
)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [
    language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) 
    for i in range(4)
]
validation_char_ngram_lms = [
    language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) 
    for i in range(4)
]
for i in range(4):
    print("validation set JSD for n={}: {}".format(
        i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])
    ))
true_char_ngram_lms = [
    language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) 
    for i in range(4)
]

def generate_samples():
    samples = Generator(BATCH_SIZE)
    samples = tf.argmax(samples, axis=2).numpy()
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples

gen = inf_train_gen()

for iteration in range(ITERS):
    start_time = time.time()

    # Train generator
    if iteration > 0:
        _ = train_generator()

    # Train critic
    for i in range(CRITIC_ITERS):
        _data = next(gen)
        _disc_cost = train_discriminator(_data)

    lib.plot.plot('time', time.time() - start_time)
    lib.plot.plot('train disc cost', _disc_cost.numpy())

    if iteration % 100 == 99:
        checkpoint_manager.save(checkpoint_number=iteration)
        samples = []
        for i in range(10):
            samples.extend(generate_samples())

        for i in range(4):
            lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
            lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

        with open('samples_{}.txt'.format(iteration), 'w') as f:
            for i, s in enumerate(samples):
                s = "".join(s)
                f.write('>' + str(i) + '\n')
                f.write(s + "\n")

    if iteration % 100 == 99:
        lib.plot.flush()
    
    lib.plot.tick()
