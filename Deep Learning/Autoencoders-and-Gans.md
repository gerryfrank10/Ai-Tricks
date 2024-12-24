# AutoEncoders

## 1. What are Autoencoders?
Autoencoders are **unsupervised neural networks** designed to reconstruct their input. They efficiently compress and reconstruct data, acting as dimensionality reduction and feature extraction architectures.

### Key Components:
1. **Encoder**: Compresses the input into a latent-space representation (bottleneck).
2. **Latent Space**: The compressed, lower-dimensional representation (features).
3. **Decoder**: Reconstructs the input from the latent-space representation.

### Use Cases:
- Dimensionality reduction.
- Anomaly detection.
- Denoising (removing noise from data).
- Image generation.

---

### 2. Libraries for AutoEncoders
1. **TensorFlow/Keras**:
   - Simplifies autoencoder implementation.
   ```bash
   pip install tensorflow
   ```

2. **PyTorch**:
   - Preferred for custom autoencoder models.
   ```bash
   pip install torch torchvision
   ```

3. **scikit-learn**:
   - For feature extraction or simpler workflows.

---

### 3. AutoEncoder Implementation in TensorFlow/Keras

#### **3.1 Defining the Architecture**
```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define dimensions
input_dim = 784   # For flattened 28x28 images (e.g., MNIST)
encoding_dim = 64  # Size of the latent space

# Define layers
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

# Model definitions
autoencoder = Model(input_img, decoded)  # Overall autoencoder
encoder = Model(input_img, encoded)      # Encoder for feature extraction
```

#### **3.2 Compiling and Training**
```python
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Training with dummy data
import numpy as np
x_train = np.random.rand(10000, 784)  # Replace with real data
x_train_noisy = x_train + 0.2 * np.random.normal(size=x_train.shape)  # Add noise for denoising

autoencoder.fit(x_train_noisy, x_train, epochs=50, batch_size=256, shuffle=True)
```

#### **3.3 Extracting Latent Features**
```python
encoded_imgs = encoder.predict(x_train)  # Latent features
```

---

### 4. Denoising AutoEncoder
Autoencoders can remove noise from datasets by learning a cleaner reconstruction.

```python
# Adding Gaussian noise to input images
noisy_data = x_train + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
noisy_data = np.clip(noisy_data, 0., 1.)  # Clipping to keep values in range

# Train autoencoder
autoencoder.fit(noisy_data, x_train, epochs=20, batch_size=128, shuffle=True)

# Reconstruct cleaned images
cleaned_data = autoencoder.predict(noisy_data)
```

---

### 5. Variational AutoEncoders (VAEs)
VAEs extend classical autoencoders by introducing a **probabilistic latent space**, useful for generating new data points.

```python
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# Custom sampling layer for latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))  # Random noise
    return z_mean + K.exp(z_log_var / 2) * epsilon
```

The key difference in VAEs is their ability to **generate new samples** by sampling from a latent distribution.

---

```markdown
# GANs (Generative Adversarial Networks)

## 1. What are GANs?
**Generative Adversarial Networks (GANs)** consist of two networks:
1. **Generator**: Creates synthetic data resembling the training data.
2. **Discriminator**: Evaluates the authenticity of data samples (real vs. fake).

The two networks compete:
- The **generator** tries to improve until the discriminator can't distinguish between real and fake data.

### Use Cases:
- Image generation (e.g., DeepFake creation).
- Text and video generation.
- Data augmentation.
- Super-resolution.

---

## 2. Libraries for GANs
1. **TensorFlow/Keras**:
   Simplifies custom GAN training workflows.
2. **PyTorch**:
   Provides raw flexibility, suitable for deep customization.

```bash
pip install tensorflow
pip install torch torchvision
```

---

## 3. GAN Architecture and Workflow

### **3.1 How GANs Work**
1. The **Generator**:
   - Takes a random vector as input and generates synthetic data.
   
2. The **Discriminator**:
   - Takes real or generated data as input and predicts whether it’s real or fake.

3. The **Adversarial Process**:
   - Generator minimizes its loss: improving at “fooling” the discriminator.
   - Discriminator maximizes its accuracy: improving at spotting fakes.

---

### 4. Simple GAN Implementation in Keras

#### **Step 1: Import Required Libraries**
```python
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
```

---

#### **Step 2: Define the Generator**
```python
# Generator creates "fake data" from random noise
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(28 * 28, activation='tanh'))  # Output shape for MNIST
    model.add(BatchNormalization(momentum=0.8))
    
    return model

latent_dim = 100
generator = build_generator(latent_dim)
```

---

#### **Step 3: Define the Discriminator**
```python
from tensorflow.keras.optimizers import Adam

# Discriminator decides if an input is real or fake
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=784))  # Flattened 28x28 MNIST inputs
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (real/fake)
    
    model.compile(optimizer=Adam(0.0002), loss='binary_crossentropy', metrics=['accuracy'])
    return model

discriminator = build_discriminator()
```

---

#### **Step 4: Combine Generator and Discriminator**
```python
# GAN combines generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    
    return Model(gan_input, gan_output)

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')
```

---

#### **Step 5: Training GANs**
```python
# Training loop
def train_gan(generator, discriminator, gan, epochs, batch_size):
    for epoch in range(epochs):
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        
        # Get real images (for example, from MNIST dataset)
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
        
        # Train discriminator on real and fake samples
        real_labels = np.ones((batch_size, 1))  # Real images labeled as 1
        fake_labels = np.zeros((batch_size, 1))  # Fake images labeled as 0
        
        discriminator.train_on_batch(real_images, real_labels)
        discriminator.train_on_batch(fake_images, fake_labels)
        
        # Train GAN: Try to fool discriminator
        gan_labels = np.ones((batch_size, 1))  # Fool discriminator into thinking generated data is "real"
        gan.train_on_batch(noise, gan_labels)
```

---

## 5. Advanced GAN Variants
1. **DCGAN (Deep Convolutional GAN)**:
   - Uses convolutional layers in both generator and discriminator for image generation.

2. **CycleGAN**:
   - Translates images from one domain to another (e.g., converting summer photos to winter).

3. **WGAN (Wasserstein GAN)**:
   - Optimizes generator/discriminator training stability using a Wasserstein distance metric.

4. **Pix2Pix**:
   - Image-to-image translation (e.g., translating sketches to colored images).

---

This section covers everything from AutoEncoders to GANs, with explanations and step-by-step code examples to help you master these advanced neural network architectures.
```
