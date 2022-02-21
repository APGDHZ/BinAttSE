# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import tensorflow as tf
import numpy as np
from random import seed
import os


class _ChannelNorm(tf.keras.layers.Layer):
    def __init__(self, encoded_len, channel_size, **kwargs):
        super(_ChannelNorm, self).__init__(**kwargs)
        self.encoded_len = encoded_len
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(
            initial_value=w_init(shape=(1, 1, self.channel_size), dtype=self.dt),
            trainable=True,
            name="CN_gamma",
        )
        self.beta = tf.Variable(
            initial_value=w_init(shape=(1, 1, self.channel_size), dtype=self.dt),
            trainable=True,
            name="CN_beta",
        )

    def get_config(self):
        config = super(_ChannelNorm, self).get_config().copy()
        config.update(
            {"encoded_len": self.encoded_len, "channel_size": self.channel_size}
        )
        return config

    def call(self, inputs):
        E = tf.reshape(
            tf.reduce_mean(inputs, axis=[2], name="E_CN_rm"),
            [-1, self.encoded_len, 1],
            name="E_CN_reshape",
        )
        Var = tf.reshape(
            tf.reduce_mean((inputs - E) ** 2, axis=[2], name="Var_CN_rm"),
            [-1, self.encoded_len, 1],
            name="Var_CN_reshape",
        )
        return (
            (inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())
        ) * self.gamma + self.beta


class _GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, channel_size, **kwargs):
        super(_GlobalNorm, self).__init__(**kwargs)
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(
            initial_value=w_init(shape=(1, 1, self.channel_size), dtype=self.dt),
            trainable=True,
            name="GN_gamma",
        )
        self.beta = tf.Variable(
            initial_value=w_init(shape=(1, 1, self.channel_size), dtype=self.dt),
            trainable=True,
            name="GN_beta",
        )

    def get_config(self):
        config = super(_GlobalNorm, self).get_config().copy()
        config.update({"channel_size": self.channel_size})
        return config

    def call(self, inputs):
        E = tf.reshape(
            tf.reduce_mean(inputs, axis=[1, 2], name="E_GN_rm"),
            [-1, 1, 1],
            name="E_GN_reshape",
        )
        Var = tf.reshape(
            tf.reduce_mean((inputs - E) ** 2, axis=[1, 2], name="Var_GN_rm"),
            [-1, 1, 1],
            name="Var_GN_reshape",
        )
        return (
            (inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())
        ) * self.gamma + self.beta


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, L, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.N = N
        self.L = L

    def build(self, inputs):
        self.encoder = tf.keras.layers.Conv1D(
            filters=self.N,
            kernel_size=self.L,
            strides=self.L // 2,
            activation="linear",
            name="encode_conv1d",
        )

    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({"layers": self.encoder, "n": self.N, "l": self.L})
        return config

    def call(self, inputs, **kwargs):
        encoded_input = self.encoder(inputs=tf.expand_dims(inputs, -1))
        return encoded_input


class TCN(tf.keras.layers.Layer):
    def __init__(
        self,
        N,
        L,
        B,
        H,
        S,
        P,
        X,
        R,
        causal,
        skip,
        duration,
        sample_rate,
        encoding,
        GPU,
        **kwargs
    ):
        super(TCN, self).__init__(**kwargs)
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.S = S
        self.P = P
        self.X = X
        self.R = R
        self.GPU = GPU
        self.skip = skip
        self.causal = causal
        self.duration = duration
        self.encoding = encoding
        self.sample_rate = sample_rate

        if self.causal:
            self.padding = "causal"
        else:
            self.padding = "same"

    def build(self, inputs):

        if self.causal:
            if self.encoding == "stft":
                self.encoded_len = (
                    int(
                        np.floor(
                            (self.duration * self.sample_rate - 2 * self.N)
                            / (self.N // 2)
                        )
                    )
                    + 1
                )
                inp_norm = _ChannelNorm(
                    self.encoded_len, self.N + 1, name="Input_Channel_Norm"
                )
            else:
                self.encoded_len = (int(self.duration * self.sample_rate) - self.L) // (
                    self.L // 2
                ) + 1
                inp_norm = _ChannelNorm(
                    self.encoded_len, self.N, name="Input_Channel_Norm"
                )
            frst_norm = _ChannelNorm(
                self.encoded_len, self.H, name="First_Channel_Norm"
            )
            scnd_norm = _ChannelNorm(
                self.encoded_len, self.H, name="Second_Channel_Norm"
            )
        else:
            inp_norm = _GlobalNorm(self.N, name="Input_Global_Norm")
            frst_norm = _GlobalNorm(self.H, name="First_Global_Norm")
            scnd_norm = _GlobalNorm(self.H, name="Second_Global_Norm")

        layers = {
            "input_norm": inp_norm,
            "bottleneck": tf.keras.layers.Conv1D(self.B, 1, 1, name="bottleneck"),
        }

        for r in range(self.R):
            for x in range(self.X):
                this_block = "block_{}_{}_".format(r, x)

                layers[this_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1, name="first_1x1_conv"
                )

                layers[this_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="first_PReLU"
                )

                layers[this_block + "first_norm"] = frst_norm

                if self.GPU:
                    layers[this_block + "dw_conv"] = tf.keras.layers.Conv1D(
                        self.H,
                        self.P,
                        padding=self.padding,
                        dilation_rate=2 ** x,
                        groups=self.H,
                    )

                layers[this_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="second_PReLU"
                )

                layers[this_block + "second_norm"] = scnd_norm

                layers[this_block + "res_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name="res_1x1_conv"
                )

                layers[this_block + "skip_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.S, kernel_size=1, name="skip_1x1_conv"
                )

        self.lrs = layers

    def get_config(self):
        config = super(TCN, self).get_config().copy()
        config.update(
            {
                "layers": self.lrs,
                "n": self.N,
                "l": self.L,
                "b": self.B,
                "h": self.H,
                "s": self.S,
                "p": self.P,
                "x": self.X,
                "r": self.R,
                "gpu": self.GPU,
                "c": self.causal,
                "skip": self.skip,
                "sr": self.sample_rate,
                "duration": self.duration,
                "enc_len": self.encoded_len,
            }
        )
        return config

    def call(self, encoded_input, **kwargs):

        norm_input = self.lrs["input_norm"](encoded_input)

        block_input = self.lrs["bottleneck"](norm_input)

        skip_connections = 0

        for r in range(self.R):

            for x in range(self.X):

                now_block = "block_{}_{}_".format(r, x)

                block_output = self.lrs[now_block + "first_1x1_conv"](block_input)

                block_output = self.lrs[now_block + "first_PReLU"](block_output)

                block_output = self.lrs[now_block + "first_norm"](block_output)

                if self.GPU:
                    block_output = self.lrs[now_block + "dw_conv"](block_output)

                block_output = self.lrs[now_block + "second_PReLU"](block_output)

                block_output = self.lrs[now_block + "second_norm"](block_output)

                block_output = self.lrs[now_block + "res_1x1_conv"](block_output)

                skip = self.lrs[now_block + "skip_1x1_conv"](block_output)

                block_input = block_output + block_input

                skip_connections = skip_connections + skip

        if self.skip:
            return skip_connections
        else:
            return block_input


class Masker(tf.keras.layers.Layer):
    def __init__(self, N, encoding, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.encoding = encoding
        if self.encoding == "stft":
            self.N = N + 1
            self.activation = tf.keras.activations.sigmoid
        else:
            self.N = N
            self.activation = tf.keras.activations.softmax

    def build(self, inputs):
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1], name="decode_PReLU")
        self.decode = tf.keras.layers.Conv1D(self.N, 1, 1, name="1x1_conv_decoder")


    def get_config(self):
        config = super(Masker, self).get_config().copy()
        config.update(
            {
                "n": self.N,
                "prelu": self.prelu,
                "decode": self.decode,
                "activation": self.activation,
            }
        )
        return config

    def call(self, inputs, encoded_input, **kwargs):
        inputs = self.prelu(inputs)
        sep_output = self.decode(inputs)
        mask = self.activation(sep_output)
        return mask * encoded_input


class Decoder(tf.keras.layers.Layer):
    def __init__(self, L, oa, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.L = L
        self.oa = oa

    def build(self, inputs):
        self.deconv = tf.keras.layers.Conv1DTranspose(
            1,
            self.L,
            strides=self.L // 2,
            use_bias=False,
            name="1d_deconv",
            activation=self.oa,
        )

    def get_config(self):
        config = super(Decoder, self).get_config().copy()
        config.update({"l": self.L, "deconv": self.deconv})
        return config

    def call(self, inputs, **kwargs):
        output = tf.squeeze(self.deconv(inputs), -1)
        return output


class stftLayer(tf.keras.layers.Layer):
    def __init__(self, blockLen, **kwargs):
        super(stftLayer, self).__init__(**kwargs)
        self.blockLen = blockLen * 2
        self.block_shift = blockLen // 2
        self.wf = tf.signal.hamming_window

    def get_config(self):
        config = super(stftLayer, self).get_config().copy()
        config.update({"block_len": self.blockLen, "block_shift": self.block_shift})
        return config

    def call(self, inputs):
        x = tf.signal.stft(
            inputs,
            self.blockLen,
            self.block_shift,
            window_fn=self.wf,
        )

        mag = tf.math.abs(x)
        phase = tf.math.angle(x)
        return [mag, phase]


class istftLayer(tf.keras.layers.Layer):
    def __init__(self, blockLen, **kwargs):
        super(istftLayer, self).__init__(**kwargs)
        self.blockLen = blockLen * 2
        self.block_shift = blockLen // 2
        self.wf = tf.signal.hamming_window

    def get_config(self):
        config = super(istftLayer, self).get_config().copy()
        config.update({"block_len": self.blockLen, "block_shift": self.block_shift})
        return config

    def call(self, mag, phase):
        x = tf.complex(mag * tf.math.cos(phase), mag * tf.math.sin(phase))
        y = tf.signal.inverse_stft(
            x,
            self.blockLen,
            self.block_shift,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.block_shift, forward_window_fn=self.wf
            ),
        )
        return y


class Model:
    def __init__(self, args):

        os.environ["PYTHONHASHSEED"] = str(42)
        seed(42)
        tf.random.set_seed(42)
        self.N = args.N
        self.L = args.L
        self.B = args.B
        self.H = args.H
        self.S = args.S
        self.P = args.P
        self.X = args.X
        self.R = args.R
        self.GPU = args.GPU
        self.skip = args.skip
        self.top = args.topology
        self.causal = args.causal
        self.duration = args.duration
        self.encoding = args.encoding
        self.oa = args.output_activation
        self.sample_rate = args.sample_rate
        self.model_name = self.top + "_auto_encoder"

        self.stft_left = stftLayer(self.N, name="STFT_left")

        self.stft_right = stftLayer(self.N, name="STFT_right")

        self.encoder_left = Encoder(self.N, self.L, name="Encoder_left")

        self.encoder_right = Encoder(self.N, self.L, name="Encoder_right")

        self.encoder_left_aux = Encoder(self.N, self.L, name="Encoder_left_aux")

        self.encoder_right_aux = Encoder(self.N, self.L, name="Encoder_right_aux")

        self.TCN_left = TCN(
            self.N,
            self.L,
            self.B,
            self.H,
            self.S,
            self.P,
            self.X,
            self.R,
            self.causal,
            self.skip,
            self.duration,
            self.sample_rate,
            self.encoding,
            self.GPU,
            name="TCN_left",
        )

        self.TCN_right = TCN(
            self.N,
            self.L,
            self.B,
            self.H,
            self.S,
            self.P,
            self.X,
            self.R,
            self.causal,
            self.skip,
            self.duration,
            self.sample_rate,
            self.encoding,
            self.GPU,
            name="TCN_right",
        )

        self.attention1 = tf.keras.layers.Multiply(name="Attention_layer1")

        self.attention2 = tf.keras.layers.Multiply(name="Attention_layer2")

        self.masker_left = Masker(self.N, self.encoding, name="Masker_left")

        self.masker_right = Masker(self.N, self.encoding, name="Masker_right")

        self.decoder_left = Decoder(self.L, self.oa, name="Decoder_left")

        self.decoder_right = Decoder(self.L, self.oa, name="Decoder_right")

        self.istft_left = istftLayer(self.N, name="iSTFT_left")

        self.istft_right = istftLayer(self.N, name="iSTFT_right")

    def call(self):

        input_left = tf.keras.Input(shape=(None,), name="Input_left")

        input_right = tf.keras.Input(shape=(None,), name="Input_right")

        if self.top == "bilateral_base":

            if self.encoding == "stft":
                [enc_inp_l, phase_l] = self.stft_left(input_left)

                [enc_inp_r, phase_r] = self.stft_right(input_right)

            else:
                enc_inp_l = self.encoder_left(input_left)

                enc_inp_r = self.encoder_right(input_right)

            skp_l = self.TCN_left(enc_inp_l)

            skp_r = self.TCN_right(enc_inp_r)

            masked_left = self.masker_left(skp_l, enc_inp_l)

            masked_right = self.masker_right(skp_r, enc_inp_r)

            if self.encoding == "stft":
                out_left = self.istft_left(masked_left, phase_l)

                out_right = self.istft_right(masked_right, phase_r)

            else:
                out_left = self.decoder_left(masked_left)

                out_right = self.decoder_right(masked_right)

        elif self.top == "Double_attention":

            if self.encoding == "stft":
                [enc_inp_l, phase_l] = self.stft_left(input_left)

                [enc_inp_r, phase_r] = self.stft_right(input_right)

            else:
                enc_inp_l = self.encoder_left(input_left)

                enc_inp_r = self.encoder_right(input_right)

            kernel1 = self.attention2([enc_inp_l, enc_inp_r])

            skp_l = self.TCN_left(kernel1)

            skp_r = self.TCN_right(kernel1)

            kernel2 = self.attention1([skp_l, skp_r])

            masked_left = self.masker_left(kernel2, enc_inp_l)

            masked_right = self.masker_right(kernel2, enc_inp_r)

            if self.encoding == "stft":
                out_left = self.istft_left(masked_left, phase_l)

                out_right = self.istft_right(masked_right, phase_r)

            else:
                out_left = self.decoder_left(masked_left)

                out_right = self.decoder_right(masked_right)

        else:

            if self.encoding == "stft":
                [enc_inp_l, phase_l] = self.stft_left(input_left)

                [enc_inp_r, phase_r] = self.stft_right(input_right)

            else:
                enc_inp_l = self.encoder_left(input_left)

                enc_inp_r = self.encoder_right(input_right)

            skp_l = self.TCN_left(enc_inp_l)

            skp_r = self.TCN_right(enc_inp_r)

            kernel = self.attention1([skp_l, skp_r])

            masked_left = self.masker_left(kernel, enc_inp_l)

            masked_right = self.masker_right(kernel, enc_inp_r)

            if self.encoding == "stft":
                out_left = self.istft_left(masked_left, phase_l)

                out_right = self.istft_right(masked_right, phase_r)

            else:
                out_left = self.decoder_left(masked_left)

                out_right = self.decoder_right(masked_right)

        model = tf.keras.Model(
            inputs=[input_left, input_right],
            outputs=[out_left, out_right],
            name=self.model_name,
        )

        for i, w in enumerate(model.weights):
            split_name = w.name.split("/")
            if len(split_name) > 1:
                new_name = (
                    split_name[0]
                    + "_"
                    + str(i)
                    + "/"
                    + split_name[1]
                    + "_"
                    + str(i)
                    + "/"
                    + split_name[-1]
                )
                model.weights[i]._handle_name = new_name
            else:
                new_name = (
                    "new"
                    + "_"
                    + str(i)
                    + "/"
                    + "layer_name"
                    + "_"
                    + str(i)
                    + "/"
                    + split_name[-1]
                )

        return model


class SISDR(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SISDR, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        def calc_sdr(s_hat, s):
            def norm(x):
                return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)

            s_target = tf.reduce_sum(s_hat * s, axis=-1, keepdims=True) * s / norm(s)
            num = norm(s_target)
            den = norm(s_hat - s_target)
            return 10.0 * tf.experimental.numpy.log10(num / den)

        sdr = calc_sdr(y_pred, y_true)
        return tf.reduce_mean(-sdr)


class SNR(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SNR, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        num = tf.norm(y_true, 2, axis=-1, keepdims=True)
        den = tf.norm(y_pred - y_true, 2, axis=-1, keepdims=True)

        snr = 10.0 * tf.experimental.numpy.log10(num / den)
        return -tf.reduce_mean(snr)
