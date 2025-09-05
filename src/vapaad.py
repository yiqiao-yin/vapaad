import time
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def sequence_ce_sum(y_true, y_pred):
    """Paper-style summed cross-entropy per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Get the actual number of dimensions to handle both 4D and 5D cases
    ndims = len(bce.shape)
    if ndims == 4:  # (batch, time, height, width)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3, 4])
    return tf.reduce_mean(ce_sum_per_seq)


def mse_seq(y_true, y_pred):
    """MSE per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    se = tf.square(y_pred - y_true)
    # Get the actual number of dimensions
    ndims = len(se.shape)
    if ndims == 4:  # (batch, time, height, width)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3, 4])
    return tf.reduce_mean(mse_per_seq)


class SelfAttention(layers.Layer):
    """
    A custom self-attention layer that computes attention scores to enhance model performance by focusing on relevant parts of the input data.

    This layer creates query, key, and value representations of the input, then calculates attention scores to determine how much focus to put on each part of the input data. The output is a combination of the input and the attention mechanism's weighted focus, which allows the model to pay more attention to certain parts of the data.

    Attributes:
        query_dense (keras.layers.Dense): A dense layer for transforming the input into a query tensor.
        key_dense (keras.layers.Dense): A dense layer for transforming the input into a key tensor.
        value_dense (keras.layers.Dense): A dense layer for transforming the input into a value tensor.
        combine_heads (keras.layers.Dense): A dense layer for combining the attention heads' outputs.
    """

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, ...]):
        """
        Initializes the internal dense layers based on the last dimension of the input shape, setting up the query, key, value, and combine heads layers.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input tensor to the layer.
        """
        self.query_dense = layers.Dense(units=input_shape[-1])
        self.key_dense = layers.Dense(units=input_shape[-1])
        self.value_dense = layers.Dense(units=input_shape[-1])
        self.combine_heads = layers.Dense(units=input_shape[-1])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the self-attention mechanism on the input tensor and returns the combined output with a residual connection.

        Args:
            inputs (tf.Tensor): The input tensor to the self-attention layer.

        Returns:
            tf.Tensor: The output tensor after applying self-attention and combining with the input tensor through a residual connection.
        """
        # Generate query, key, value tensors
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Calculate attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        distribution = tf.nn.softmax(scores)
        attention_output = tf.matmul(distribution, value)

        # Combine heads and add residual connection
        combined_output = self.combine_heads(attention_output) + inputs
        return combined_output


class VAPAAD:
    """
    The VAPAAD model, short for Vision Augmentation Prediction Autoencoder with Attention Design,
    is a sophisticated neural network architecture tailored for video processing tasks. This model
    leverages a dual-encoder structure to process sequences of video frames for tasks such as
    video frame prediction and unsupervised learning. The unique aspect of this architecture is its
    stop gradient design, which effectively separates the learning phases of the two encoders, allowing
    one encoder to stabilize while the other continues to adapt during training.

    This architecture integrates data augmentation directly into the video processing pipeline,
    enhancing the model's ability to generalize across varied video data. It employs self-attention
    mechanisms to capture long-range dependencies within the video sequences, thereby enhancing its
    predictive capabilities.

    Attributes:
        input_shape (Tuple[int, int, int]): The shape of the input frames expected by the model.
        gen_main (keras.Model): The main generator model that processes the input frames.
        gen_aux (keras.Model): An auxiliary generator used to predict future frames.
        instructor (keras.Model): The instructor model that evaluates the generated frames.
        cross_entropy (tf.keras.losses.Loss): Loss function used for training the model.
        generator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator models.
        instructor_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the instructor model.

    The VAPAAD model is developed by Yiqiao Yin, who can be reached at eagle0504@gmail.com for further
    inquiries or support related to this implementation.

    Example usage:
        # Initializing a new VAPAAD model
        vapaad_model = VAPAAD(input_shape=(19, 64, 64, 1))

        # Assuming x_train and y_train are already defined and loaded
        vapaad_model.train(x_train, y_train, batch_size=32)
    """
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        # Initialize generator and instructor models
        self.gen_main = self.build_generator()
        self.gen_aux = self.build_generator()
        self.instructor = self.build_instructor()
        # Define loss functions and optimizers
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.learning_rate = tf.Variable(1e-4, trainable=False)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.instructor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def build_generator(self) -> keras.Model:
        """
        Constructs the generator model for video processing with data augmentation and self-attention.

        This method is responsible for creating a generator model that performs augmentations on input
        frames and then processes them through ConvLSTM2D layers with self-attention, finally applying a
        convolution across the time dimension to generate output frames.

        The model is part of a generative approach and could be used in tasks such as video frame prediction,
        unsupervised learning, or as a part of a Generative Adversarial Network (GAN).

        Returns:
            A Keras model that takes a sequence of frames as input, augments them via random zooming, rotations,
            and translations, and then outputs processed frames with the same sequence length as the input.

        Note: 'input_shape' should be an attribute of the class instance, and 'SelfAttention' is expected
        to be either a predefined layer in Keras or a custom implementation provided in the code.

        Example usage:
            generator = build_generator()
        """
        # Data augmentation layers intended to increase robustness and generalization
        data_augmentation = keras.Sequential(
            [
                # layers.RandomZoom(height_factor=0.00002, width_factor=0.00002),
                # layers.RandomRotation(factor=0.02),
                layers.RandomTranslation(height_factor=0.02, width_factor=0.02),
            ],
            name="data_augmentation",
        )

        # Input layer defining the shape of the input frames
        inp = layers.Input(shape=self.input_shape)
        # Apply time distributed data augmentation which applies the augmentation to each frame independently
        x = layers.TimeDistributed(data_augmentation)(inp)
        # Convolutional LSTM layer with relu activation to capture temporal features
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        # Batch normalization to help maintain the stability of the network
        x = layers.BatchNormalization()(x)
        # Self-attention layer for capturing long-range dependencies within the sequences
        x = SelfAttention()(x)
        # Conv3D layer to process the features obtained from previous layers and produce a sequence of frames
        x = layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        )(x)

        # Construct the model with the specified input and output tensors
        return keras.models.Model(inputs=inp, outputs=x)

    def build_instructor(self) -> keras.Model:
        """
        Constructs the instructor model with convolutional LSTM and fully connected layers.

        This method specifically builds a video processing instructor model that uses ConvLSTM2D layers,
        followed by self-attention, global average pooling, and dense layers to process the input frames
        and predict a one-dimensional output.

        The architecture is designed for sequential data processing ideal for video or time-series data.

        Returns:
            A compiled Keras model that takes a sequence of frames as input and outputs a
            one-dimensional tensor after processing through ConvLSTM2D, self-attention,
            and dense layers. The output can be interpreted as the probability of a certain
            class or a value depending on the final activation function used (sigmoid in this case).

        Note: 'input_shape' should be an attribute of the class instance, and 'SelfAttention' is
        assumed to be a pre-defined layer or a custom layer implemented elsewhere in the code.

        Example usage:
            model = build_instructor()
        """
        # Input layer defining the shape of the input frames
        inp = layers.Input(shape=self.input_shape)
        # Convolutional LSTM layer with relu activation
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        # Batch Normalization layer
        x = layers.BatchNormalization()(x)
        # Self-attention layer for sequence learning
        x = SelfAttention()(x)
        # Global Average Pooling across the frames to get a feature vector
        x = layers.GlobalAveragePooling3D()(x)
        # Fully connected layers with relu activation
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        # Output layer with sigmoid activation for binary classification or regression tasks
        output = layers.Dense(1, activation="sigmoid")(x)

        # Construct the model with specified layers
        return keras.models.Model(inputs=inp, outputs=output)

    def update_learning_rate(self, current_loss: float, previous_loss: float, decay_rate: float = 0.9) -> None:
        """
        Adjusts the learning rate based on the comparison between current and previous loss.

        The learning rate is reduced by 10% if the current loss is greater than the previous loss,
        ensuring that the learning rate does not fall below a minimum threshold of 1e-6.
        """
        # Check if the current loss has increased compared to the previous loss
        if current_loss > previous_loss:
            # Calculate the new learning rate by reducing the current learning rate by 10%
            current_lr = self.learning_rate.numpy()
            new_lr = current_lr * decay_rate
            # Update the learning rate with the higher value between the new learning rate and the minimum threshold
            new_lr = max(new_lr, 1e-6)
            self.learning_rate.assign(new_lr)
            
            # Update the optimizers' learning rates
            self.generator_optimizer.learning_rate.assign(new_lr)
            self.instructor_optimizer.learning_rate.assign(new_lr)

    def train_step(
        self, images: tf.Tensor, future_images: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform a single training step by updating the generator and instructor models.

        This method applies gradient descent to both the generator and the instructor models
        based on the loss computed from the real and generated images.

        Args:
            images (tf.Tensor): A tensor of input images for the current time step provided
                                to the generator model 'gen_main'.
            future_images (tf.Tensor): A tensor of target images for the future time step provided
                                    to the generator model 'gen_aux'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing the loss values for 
                                        the generator model ('gen_loss'), instructor model ('inst_loss'),
                                        custom sequence cross-entropy sum ('seq_ce'), and MSE per sequence ('mse_seq').

        Note: 'gen_optimizer' and 'inst_optimizer' should be attributes of the class instance.

        The function uses TensorFlow operations and assumes that 'gen_main', 'gen_aux', 'instructor',
        'generator_optimizer', 'instructor_optimizer', 'generator_loss', and 'instructor_loss' are
        defined as attributes of the class in which this method is implemented.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as inst_tape:
            # Generate outputs for both current and future inputs
            output_main = self.gen_main(images, training=True)
            output_aux = self.gen_aux(future_images, training=True)
            real_output = self.instructor(output_aux, training=True)
            fake_output = self.instructor(output_main, training=True)

            # Calculate losses for both models with custom metrics
            gen_loss = self.generator_loss(fake_output, output_main, future_images)
            inst_loss = self.instructor_loss(real_output, fake_output, future_images, output_main)

        # Apply gradients to update model weights
        gradients_of_gen = gen_tape.gradient(
            gen_loss, self.gen_main.trainable_variables
        )
        gradients_of_inst = inst_tape.gradient(
            inst_loss, self.instructor.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_gen, self.gen_main.trainable_variables)
        )
        self.instructor_optimizer.apply_gradients(
            zip(gradients_of_inst, self.instructor.trainable_variables)
        )

        # Calculate custom metrics for monitoring
        seq_ce = sequence_ce_sum(future_images, output_main)
        mse_seq_val = mse_seq(future_images, output_main)

        return gen_loss, inst_loss, seq_ce, mse_seq_val

    def generator_loss(self, fake_output, generated_images, target_images):
        """
        Calculates the combined loss for the generator model including custom metrics.

        The loss encourages the generator to produce images that the instructor model classifies as real,
        while also optimizing for custom sequence-based metrics.

        Args:
        fake_output (tf.Tensor): The generator model's output logits for generated (fake) images.
        generated_images (tf.Tensor): The actual generated images from gen_main.
        target_images (tf.Tensor): The target images for comparison.

        Returns:
        tf.Tensor: The combined loss for the generator model.
        """
        # Original generator loss
        adversarial_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        
        # Custom metrics
        seq_ce_loss = sequence_ce_sum(target_images, generated_images)
        mse_seq_loss = mse_seq(target_images, generated_images)
        
        # Combined loss with scaling
        combined_loss = adversarial_loss + 0.001 * seq_ce_loss + 1.0 * mse_seq_loss
        
        return combined_loss

    def instructor_loss(self, real_output, fake_output, real_images, generated_images):
        """
        Calculates the combined loss for the instructor model including custom metrics.

        The loss is computed as the sum of the cross-entropy losses for the real and fake outputs,
        plus custom sequence-based metrics to improve discrimination quality.

        Args:
        real_output (tf.Tensor): The instructor model's output logits for real images.
        fake_output (tf.Tensor): The instructor model's output logits for generated (fake) images.
        real_images (tf.Tensor): The actual real/target images.
        generated_images (tf.Tensor): The generated images from gen_main.

        Returns:
        tf.Tensor: The combined loss for the instructor model.
        """
        # Original instructor loss
        real_loss = self.cross_entropy(tf.zeros_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        adversarial_loss = real_loss + fake_loss
        
        # Custom metrics to improve discrimination
        seq_ce_loss = sequence_ce_sum(real_images, generated_images)
        mse_seq_loss = mse_seq(real_images, generated_images)
        
        # Combined loss with scaling
        combined_loss = adversarial_loss + 0.001 * seq_ce_loss + 1.0 * mse_seq_loss
        
        return combined_loss

    def train(self, x_train, y_train, batch_size=64, epochs=1):
        """
        Trains the model for a specified number of epochs and batch size.

        This function iterates over the entire dataset for the specified number of epochs,
        randomly selecting batches of data to perform training steps. The selection is random
        and without replacement within each epoch, ensuring diverse exposure of data.

        Args:
        x_train (np.ndarray): The input training data.
        y_train (np.ndarray): The target training data.
        batch_size (int, optional): The number of samples per batch of computation. Defaults to 64.
        epochs (int, optional): The number of epochs to train. Defaults to 1.

        Returns:
        None
        """
        previous_loss = float('inf')
        n_samples = x_train.shape[0]
        start = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            epoch_gen_loss = 0.0
            epoch_inst_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                if i + batch_size > n_samples:
                    continue  # Avoid index error on the last batch if it's smaller than the batch size
                selected_indices = indices[i : i + batch_size]
                x_batch = x_train[selected_indices]
                y_batch = y_train[selected_indices]
                curr_gen_loss, curr_inst_loss, curr_seq_ce, curr_mse_seq = self.train_step(x_batch, y_batch)
                
                # Accumulate losses for epoch summary
                epoch_gen_loss += float(curr_gen_loss)
                epoch_inst_loss += float(curr_inst_loss)
                n_batches += 1
                
                # if curr_gen_loss < 0.2:  # Early stopping condition
                #     print(
                #         f"> running: current sample {i + 1}, gen_loss={curr_gen_loss}, inst_loss={curr_inst_loss}, time={time.time() - start} sec"
                #     )
                #     return

                # Update learning rate based on the loss
                self.update_learning_rate(curr_gen_loss, previous_loss)
                previous_loss = curr_gen_loss

                print(
                    f"> running: epoch {epoch + 1}/{epochs}, batch {i//batch_size + 1}, gen_loss={curr_gen_loss:.4f}, inst_loss={curr_inst_loss:.4f}, seq_ce={curr_seq_ce:.4f}, mse_seq={curr_mse_seq:.4f}, time={time.time() - start:.2f} sec"
                )
            
            # Print epoch summary
            avg_gen_loss = epoch_gen_loss / n_batches if n_batches > 0 else 0
            avg_inst_loss = epoch_inst_loss / n_batches if n_batches > 0 else 0
            print(f"Epoch {epoch + 1} completed - Avg Gen Loss: {avg_gen_loss:.4f}, Avg Inst Loss: {avg_inst_loss:.4f}")

    def __read_me__(self):
        """
        This function prints a multi-line formatted instruction manual for running a VAPAAD model.

        The instructions include how to inspect the data shapes of training and validation datasets,
        initializing the VAPAAD model, selecting a random subset of the training data for training,
        and finally, running the model with GPU support if available.

        There are no parameters for this function and it doesn't return anything.
        It simply prints the instructional text to the console when called.
        """
        now = datetime.now()
        current_year = now.year
        print(
            f"""
            ## Instructions

            Assume you have data as the follows:

            ```py
            # Inspect the dataset.
            print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
            print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

            # output
            # Training Dataset Shapes: (900, 19, 64, 64, 1), (900, 19, 64, 64, 1)
            # Validation Dataset Shapes: (100, 19, 64, 64, 1), (100, 19, 64, 64, 1)
            ```

            To run the model, execute the following:
            ```py
            # Initializing a new VAPAAD model
            vapaad_model = VAPAAD(input_shape=(19, 64, 64, 1))

            # Assuming x_train and y_train are already defined and loaded
            num_samples = 64
            indices = np.random.choice(x_train.shape[0], num_samples, replace=True)
            print(indices[0:6])
            x_train_sub = x_train[indices]
            y_train_sub = y_train[indices]
            print(x_train_sub.shape, y_train_sub.shape)

            # Example usage:
            BATCH_SIZE = 3
            if tf.test.gpu_device_name() != '':
                with tf.device('/device:GPU:0'):
                    vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
            else:
                vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
            ```

            Copyright © 2010-{current_year} Present Yiqiao Yin
            """
        )
