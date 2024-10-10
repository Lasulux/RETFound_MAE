import tensorflow as tf


def add_layers(customname="naturalist3"):
    if customname == "naturalist3":
        print("Loading: ", customname)
        return [
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(1024)
        ]

    elif customname == "ala pali":
        print("Loading: ", customname)
        return [
            tf.keras.layers.Dense(2048, use_bias=False), 
            tf.keras.layers.BatchNormalization(),  # Adding Batch Normalization
            tf.keras.layers.Activation('relu'), # non-linearity
            tf.keras.layers.Dropout(0.5),  # Adding Dropout

            tf.keras.layers.Dense(1024, use_bias=False),
            tf.keras.layers.BatchNormalization(),  # Adding Batch Normalization
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),  # Adding Dropout

            tf.keras.layers.Dense(1024, use_bias=False),
            tf.keras.layers.BatchNormalization(),  # Adding Batch Normalization
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),  # Adding Dropout
        ]
    
    elif customname == "minigen":
        print("Loading: ", customname)
        return [
            tf.keras.layers.Dense(1024),
            tf.keras.layers.BatchNormalization(),  # Adding Batch Normalization
            tf.keras.layers.Activation('relu'), # non-linearity
            tf.keras.layers.Dropout(0.3),  # Adding Dropout       
            #  Mean Absolute Error: 7.6682136326620025 (validation set, b4, 100 epochs)
        ]

    else:
        print("Not found: ", customname, "Loading default single dense")	
        return [tf.keras.layers.Dense(1024)]