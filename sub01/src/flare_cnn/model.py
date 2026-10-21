"""The two-head 1D-CNN (final design: SPLIT_W2).

    light curve (128 x 2) -> shared conv blocks 1-3 (learn the flare shape)
                          -> conv block 4 (detection)  -> "is it a flare?"  (0 to 1)
                          -> conv block 4 (energy) + star info -> tier (C/M/X/SF)

The ablation (notebook 05) showed negative transfer: with one shared top block,
the energy head did worse than an energy-only model. Giving each head its own
block 4 and weighting the energy loss 2x fixed it (notebook 06).
"""

from tensorflow import keras
from tensorflow.keras import layers

WINDOW = 128
N_CHANNELS = 2
N_STAR = 3          # Teff, log10 radius, log10 distance (scaled)


def conv_block(x, filters, kernel, pool=True):
    """Conv1D -> BatchNorm -> ReLU (-> MaxPool halves the length)."""
    x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling1D(2)(x) if pool else x


def pooled(x):
    """Average + max over time, then dropout. Max catches short peaks, average the overall shape."""
    avg = layers.GlobalAveragePooling1D()(x)
    mx = layers.GlobalMaxPooling1D()(x)
    return layers.Dropout(0.3)(layers.Concatenate()([avg, mx]))


def build_model(tier_weight=2.0, split_last=True, learning_rate=1e-3):
    """Build and compile the two-head CNN.

    Defaults = final model v2 (SPLIT_W2): separate block 4 per head, energy loss x2.
    split_last=False, tier_weight=1.0 gives the original v1 design.
    """
    lc_in = keras.Input((WINDOW, N_CHANNELS), name="light_curve")
    star_in = keras.Input((N_STAR,), name="star_info")

    # Shared encoder
    x = conv_block(lc_in, 32, 7)
    x = conv_block(x, 64, 5)
    x = conv_block(x, 128, 5)

    # Top block: separate per head (v2) or shared (v1)
    if split_last:
        f_det = pooled(conv_block(x, 128, 3, pool=False))
        f_tier = pooled(conv_block(x, 128, 3, pool=False))
    else:
        f_det = f_tier = pooled(conv_block(x, 128, 3, pool=False))

    # Head 1: detection
    det = layers.Dense(64, activation="relu")(f_det)
    det_out = layers.Dense(1, activation="sigmoid", name="flare")(det)

    # Head 2: energy tier (light-curve features + star info)
    s = layers.Dense(16, activation="relu")(star_in)
    e = layers.Concatenate()([f_tier, s])
    e = layers.Dropout(0.3)(layers.Dense(64, activation="relu")(e))
    tier_out = layers.Dense(4, activation="softmax", name="tier")(e)

    model = keras.Model({"light_curve": lc_in, "star_info": star_in},
                        {"flare": det_out, "tier": tier_out})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={"flare": "binary_crossentropy", "tier": "sparse_categorical_crossentropy"},
        loss_weights={"flare": 1.0, "tier": tier_weight},
    )
    return model


def load_model(path):
    """Load a saved .keras model (for example models/best.keras)."""
    return keras.models.load_model(path)
