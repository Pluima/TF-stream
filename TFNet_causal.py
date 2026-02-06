import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _twoch_to_complex(two_ch):
    """(B, 2, F, T) -> (B, F, T) complex"""
    real = two_ch[:, 0]
    imag = two_ch[:, 1]
    # torch.complex does not accept bfloat16
    if real.dtype == torch.bfloat16:
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


def _complex_to_twoch(cpx):
    """(B, F, T) complex -> (B, 2, F, T)"""
    return torch.stack((cpx.real, cpx.imag), dim=1)


def _complex_to_twoch_multi(cpx):
    """(B, C, F, T) complex -> (B, 2*C, F, T)"""
    return torch.cat((cpx.real, cpx.imag), dim=1)


def _group_norm(channels, num_groups=8):
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CausalConv2d(nn.Module):
    """
    Causal 2D conv on time axis (T): padding only on the left side of time.
    Frequency axis uses symmetric padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        freq_pad: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if freq_pad is None:
            freq_pad = kernel_size[0] // 2
        self.freq_pad = freq_pad
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_f, k_t = self.kernel_size
        pad_t = k_t - 1
        pad_f = self.freq_pad
        x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=_group_norm):
        super().__init__()
        self.conv = nn.Sequential(
            CausalConv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1)),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
            CausalConv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1)),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.conv(x)


def causal_unet_conv(input_nc, output_nc, norm_layer=_group_norm):
    downconv = CausalConv2d(
        input_nc, output_nc, kernel_size=(4, 4), stride=(2, 1), freq_pad=1
    )
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


class CausalUpConv(nn.Module):
    def __init__(self, ch_in, ch_out, outermost=False, norm_layer=_group_norm, scale=(2, 1)):
        super().__init__()
        self.scale = scale
        conv = CausalConv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1))
        if outermost:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale),
                conv,
                nn.Sigmoid(),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale),
                conv,
                norm_layer(ch_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.up(x)


def causal_unet_upconv(input_nc, output_nc, outermost=False, norm_layer=_group_norm, scale=(2, 1)):
    upsample = nn.Upsample(scale_factor=scale)
    conv = CausalConv2d(input_nc, output_nc, kernel_size=(3, 3), stride=(1, 1))
    if outermost:
        return nn.Sequential(*[upsample, conv])
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    return nn.Sequential(*[upsample, conv, upnorm, uprelu])


class CausalUnet(nn.Module):
    """
    Causal U-Net for mask prediction with time-causal convolutions.
    """

    def __init__(
        self,
        ngf=64,
        input_nc=2,
        output_nc=2,
        cond_dim=128,
        vec_dim_per_source=3,
        bottleneck_blocks=2,
        bottleneck_residual=True,
    ):
        super().__init__()
        self.audionet_convlayer1 = causal_unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = causal_unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = CausalConvBlock(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = CausalConvBlock(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = CausalConvBlock(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = CausalConvBlock(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = CausalConvBlock(ngf * 8, ngf * 8)
        self.audionet_convlayer8 = CausalConvBlock(ngf * 8, ngf * 8)
        self.frequency_pool = nn.MaxPool2d([2, 1])

        self.bottleneck_residual = bottleneck_residual
        self.bottleneck_blocks = nn.ModuleList(
            [CausalConvBlock(ngf * 8, ngf * 8) for _ in range(max(0, bottleneck_blocks))]
        )

        self.DirVecNet = nn.Sequential(
            nn.Linear(vec_dim_per_source, cond_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(cond_dim, cond_dim),
        )

        self.audionet_upconvlayer1 = CausalUpConv(cond_dim + ngf * 8, ngf * 8)
        self.audionet_upconvlayer2 = CausalUpConv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = CausalUpConv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = CausalUpConv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer5 = CausalUpConv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer6 = CausalUpConv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer7 = causal_unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer8 = causal_unet_upconv(
            ngf * 2 + cond_dim, output_nc, True
        )
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def _encode(self, audio_mix_spec):
        audio_conv1feature = self.audionet_convlayer1(audio_mix_spec[:, :, :-1, :])
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)

        audio_conv3feature = self.frequency_pool(self.audionet_convlayer3(audio_conv2feature))
        audio_conv4feature = self.frequency_pool(self.audionet_convlayer4(audio_conv3feature))
        audio_conv5feature = self.frequency_pool(self.audionet_convlayer5(audio_conv4feature))
        audio_conv6feature = self.frequency_pool(self.audionet_convlayer6(audio_conv5feature))
        audio_conv7feature = self.frequency_pool(self.audionet_convlayer7(audio_conv6feature))
        audio_conv8feature = self.frequency_pool(self.audionet_convlayer8(audio_conv7feature))
        for block in self.bottleneck_blocks:
            if self.bottleneck_residual:
                audio_conv8feature = audio_conv8feature + block(audio_conv8feature)
            else:
                audio_conv8feature = block(audio_conv8feature)

        return (
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
        )

    def _decode_single(
        self,
        audio_conv1feature,
        audio_conv2feature,
        audio_conv3feature,
        audio_conv4feature,
        audio_conv5feature,
        audio_conv6feature,
        audio_conv7feature,
        audio_conv8feature,
        emb,
    ):
        dir_map = emb.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, audio_conv8feature.size(2), audio_conv8feature.size(3)
        )
        x = torch.cat((dir_map, audio_conv8feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(x)
        audio_upconv2feature = self.audionet_upconvlayer2(
            torch.cat((audio_upconv1feature, audio_conv7feature), dim=1)
        )
        audio_upconv3feature = self.audionet_upconvlayer3(
            torch.cat((audio_upconv2feature, audio_conv6feature), dim=1)
        )
        audio_upconv4feature = self.audionet_upconvlayer4(
            torch.cat((audio_upconv3feature, audio_conv5feature), dim=1)
        )
        audio_upconv5feature = self.audionet_upconvlayer5(
            torch.cat((audio_upconv4feature, audio_conv4feature), dim=1)
        )
        audio_upconv6feature = self.audionet_upconvlayer6(
            torch.cat((audio_upconv5feature, audio_conv3feature), dim=1)
        )
        audio_upconv7feature = self.audionet_upconvlayer7(
            torch.cat((audio_upconv6feature, audio_conv2feature), dim=1)
        )

        if audio_upconv7feature.shape[2:] != audio_conv1feature.shape[2:]:
            audio_upconv7feature = F.interpolate(
                audio_upconv7feature,
                size=audio_conv1feature.shape[2:],
                mode="nearest",
            )

        dir_map = emb.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, audio_upconv7feature.size(2), audio_upconv7feature.size(3)
        )
        audio_upconv7feature = torch.cat((audio_upconv7feature, dir_map), dim=1)

        prediction = self.audionet_upconvlayer8(
            torch.cat((audio_upconv7feature, audio_conv1feature), dim=1)
        )
        return prediction

    def forward(self, audio_mix_spec, vec_feature, activation="Sigmoid", return_tuple=False):
        feats = self._encode(audio_mix_spec)
        (
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
        ) = feats

        if vec_feature.dim() == 2 and vec_feature.size(1) == 6:
            dir1 = vec_feature[:, :3]
            dir2 = vec_feature[:, 3:]
        elif vec_feature.dim() == 3 and vec_feature.size(1) == 2 and vec_feature.size(2) == 3:
            dir1 = vec_feature[:, 0, :]
            dir2 = vec_feature[:, 1, :]
        else:
            raise ValueError(f"vec_feature must be (B, 6) or (B, 2, 3), got {tuple(vec_feature.shape)}")

        emb1 = self.DirVecNet(dir1)
        emb2 = self.DirVecNet(dir2)

        pred1 = self._decode_single(
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
            emb1,
        )
        pred2 = self._decode_single(
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
            emb2,
        )

        predictions = torch.stack([pred1, pred2], dim=1)

        if activation == "Sigmoid":
            masks = self.Sigmoid(predictions)
        elif activation == "Tanh":
            masks = 4 * self.Tanh(predictions)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if return_tuple:
            return masks[:, 0], masks[:, 1]
        return masks


class CausalTFNetSeparator(nn.Module):
    """
    Strictly causal TFNet separator:
      - Causal convs on time axis
      - STFT center=False
      - Optional strictly streaming training (left context only, no future)
    """

    def __init__(self, args):
        super().__init__()
        self.stft_frame = args.network_audio.stft_frame
        self.stft_hop = args.network_audio.stft_hop
        self.n_fft = args.network_audio.n_fft
        if self.n_fft != self.stft_frame:
            # For strict causal (center=False), use n_fft == win_length to avoid
            # overlap-add zeros that cause istft errors.
            self.n_fft = self.stft_frame
        self.activation = args.network_audio.activation
        self.stereo_loss = args.stereo_loss
        window = torch.hann_window(self.stft_frame)
        window_eps = float(getattr(args.network_audio, "causal_window_eps", 1e-4))
        if window_eps > 0:
            window = window + window_eps
        self.register_buffer("window", window, persistent=False)

        input_nc = args.network_audio.input_nc
        if hasattr(args, "input_mono") and not args.input_mono:
            if input_nc == 2:
                input_nc = 4
        self.input_nc = input_nc
        output_nc = args.network_audio.output_nc

        self.mask_net = CausalUnet(
            ngf=args.network_audio.ngf,
            input_nc=input_nc,
            output_nc=output_nc,
            cond_dim=args.network_audio.cond_dim,
            bottleneck_blocks=getattr(args.network_audio, "bottleneck_blocks", 2),
            bottleneck_residual=getattr(args.network_audio, "bottleneck_residual", True),
        )

        self.streaming_train = bool(getattr(args.network_audio, "streaming_train", False))
        self.streaming_emit_len = int(getattr(args.network_audio, "streaming_emit_len", 0))
        self.streaming_left_len = int(getattr(args.network_audio, "streaming_left_len", 0))
        if self.streaming_train:
            if self.streaming_emit_len <= 0:
                raise ValueError("streaming_emit_len must be > 0 when streaming_train is enabled")
            if self.streaming_left_len < 0:
                raise ValueError("streaming_left_len must be >= 0")

    def _stft(self, audio):
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(audio.device),
            center=False,
            return_complex=True,
        )

    def _istft(self, spec, length):
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(spec.device),
            center=False,
            length=length,
        )

    def _pad_to_hop(self, audio: torch.Tensor, length: int) -> Tuple[torch.Tensor, int]:
        if length < self.n_fft:
            target_len = self.n_fft
        else:
            remainder = (length - self.n_fft) % self.stft_hop
            target_len = length if remainder == 0 else (length + (self.stft_hop - remainder))
        if target_len == length:
            return audio, length
        pad_len = target_len - length
        if audio.dim() == 3:
            pad = torch.zeros((audio.size(0), audio.size(1), pad_len), device=audio.device, dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=2)
        else:
            pad = torch.zeros((audio.size(0), pad_len), device=audio.device, dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=1)
        return audio, target_len

    def _forward_full(self, audio_mix, vec_feature, return_masks=False):
        if audio_mix.dim() == 1:
            audio_mix = audio_mix.unsqueeze(0)
        stereo_input = False
        if audio_mix.dim() == 3:
            if audio_mix.size(1) != 2:
                raise ValueError(f"stereo audio_mix must have shape (B, 2, T), got {tuple(audio_mix.shape)}")
            stereo_input = True
        elif audio_mix.dim() != 2:
            raise ValueError(f"audio_mix must be (B, T), (B, 2, T), or (T,), got {tuple(audio_mix.shape)}")
        if stereo_input and self.input_nc != 4:
            raise ValueError(f"stereo input requires input_nc=4, got input_nc={self.input_nc}")
        if not stereo_input and self.input_nc != 2:
            raise ValueError(f"mono input requires input_nc=2, got input_nc={self.input_nc}")
        B, Tlen = audio_mix.shape if not stereo_input else (audio_mix.size(0), audio_mix.size(2))
        audio_mix, padded_len = self._pad_to_hop(audio_mix, Tlen)

        if stereo_input:
            mix_cpx = torch.stack([self._stft(audio_mix[:, ch, :]) for ch in range(2)], dim=1)
        else:
            mix_cpx = self._stft(audio_mix)

        if stereo_input:
            mix_twoch = _complex_to_twoch_multi(mix_cpx)
        else:
            mix_twoch = _complex_to_twoch(mix_cpx)

        masks = self.mask_net(mix_twoch, vec_feature, activation=self.activation, return_tuple=False)

        if stereo_input:
            mix_cpx_crop = mix_cpx[:, :, :-1, :]
            target_F, target_T = mix_cpx_crop.shape[2], mix_cpx_crop.shape[3]
        else:
            mix_twoch_crop = mix_twoch[:, :, :-1, :]
            mix_cpx_crop = _twoch_to_complex(mix_twoch_crop)
            target_F, target_T = mix_twoch_crop.shape[2], mix_twoch_crop.shape[3]

        if (masks.shape[3], masks.shape[4]) != (target_F, target_T):
            Bb, S, Cc, Fm, Tm = masks.shape
            masks_4d = masks.reshape(Bb * S, Cc, Fm, Tm)
            masks_4d = F.interpolate(
                masks_4d,
                size=(target_F, target_T),
                mode="nearest",
            )
            masks = masks_4d.reshape(Bb, S, Cc, target_F, target_T)

        src_cpx_crops = []
        for src_idx in range(2):
            mask_src = masks[:, src_idx]
            if stereo_input:
                if mask_src.size(1) != 4:
                    raise ValueError(
                        f"stereo mask requires output_nc=4 (L_real,L_imag,R_real,R_imag), got {mask_src.size(1)}"
                    )
                mask_L = mask_src[:, 0:2]
                mask_R = mask_src[:, 2:4]
                mask_L_cpx = _twoch_to_complex(mask_L)
                mask_R_cpx = _twoch_to_complex(mask_R)
                src_L = mix_cpx_crop[:, 0] * mask_L_cpx
                src_R = mix_cpx_crop[:, 1] * mask_R_cpx
                src_cpx_crop = torch.stack([src_L, src_R], dim=1)
            else:
                mask_cpx = _twoch_to_complex(mask_src)
                src_cpx_crop = mix_cpx_crop * mask_cpx
            src_cpx_crops.append(src_cpx_crop)

        src_cpx_crops = torch.stack(src_cpx_crops, dim=1)

        sep_wavs = []
        for src_idx in range(2):
            src_cpx_crop = src_cpx_crops[:, src_idx]
            if stereo_input:
                src_cpx = F.pad(src_cpx_crop, (0, 0, 0, 1))
                src_wavs = [self._istft(src_cpx[:, ch], length=padded_len) for ch in range(src_cpx.size(1))]
                if self.stereo_loss:
                    src_wav = torch.stack(src_wavs, dim=1)
                else:
                    src_wav = torch.stack(src_wavs, dim=1).mean(dim=1)
            else:
                src_cpx = F.pad(src_cpx_crop, (0, 0, 0, 1))
                src_wav = self._istft(src_cpx, length=padded_len)
            sep_wavs.append(src_wav)

        sep_audio = torch.stack(sep_wavs, dim=1)
        sep_audio = sep_audio[..., :Tlen]
        if return_masks:
            return sep_audio, masks
        return sep_audio

    def _forward_streaming(self, audio_mix, vec_feature):
        if self.streaming_emit_len <= 0:
            return self._forward_full(audio_mix, vec_feature)

        if audio_mix.dim() == 1:
            audio_mix = audio_mix.unsqueeze(0)
        stereo_input = audio_mix.dim() == 3
        if stereo_input:
            if audio_mix.size(1) != 2:
                raise ValueError(f"stereo audio_mix must have shape (B, 2, T), got {tuple(audio_mix.shape)}")
            B, Tlen = audio_mix.size(0), audio_mix.size(2)
            channels = 2
        else:
            if audio_mix.dim() != 2:
                raise ValueError(f"audio_mix must be (B, T) or (B, 2, T), got {tuple(audio_mix.shape)}")
            B, Tlen = audio_mix.size(0), audio_mix.size(1)
            channels = 1

        if stereo_input and self.input_nc != 4:
            raise ValueError(f"stereo input requires input_nc=4, got input_nc={self.input_nc}")
        if not stereo_input and self.input_nc != 2:
            raise ValueError(f"mono input requires input_nc=2, got input_nc={self.input_nc}")

        hop = self.stft_hop
        win_len = self.stft_frame
        emit_len = self.streaming_emit_len
        if emit_len % hop != 0:
            raise ValueError(
                f"streaming_emit_len must be a multiple of stft_hop for true TF streaming, got "
                f"emit_len={emit_len}, hop={hop}"
            )
        emit_frames = max(1, emit_len // hop)
        left_frames = int(math.ceil(self.streaming_left_len / hop)) if self.streaming_left_len > 0 else 0
        max_frames = max(1, left_frames + emit_frames)

        audio_mix, padded_len = self._pad_to_hop(audio_mix, Tlen)
        audio_stream = audio_mix if stereo_input else audio_mix.unsqueeze(1)

        stft_buf = audio_stream.new_zeros((B, channels, 0))

        complex_dtype = torch.complex128 if audio_stream.dtype == torch.float64 else torch.complex64
        mix_cpx_buf = None
        if left_frames > 0:
            if stereo_input:
                mix_cpx_buf = torch.zeros(
                    (B, channels, self.n_fft // 2 + 1, left_frames),
                    device=audio_stream.device,
                    dtype=complex_dtype,
                )
            else:
                mix_cpx_buf = torch.zeros(
                    (B, self.n_fft // 2 + 1, left_frames),
                    device=audio_stream.device,
                    dtype=complex_dtype,
                )

        fft_dtype = torch.float32 if audio_stream.dtype in (torch.float16, torch.bfloat16) else audio_stream.dtype
        window = self.window.to(device=audio_stream.device, dtype=fft_dtype)
        window_sq = window * window

        ola_num = [torch.zeros((B, channels, win_len), device=audio_stream.device, dtype=fft_dtype) for _ in range(2)]
        ola_den = [torch.zeros((B, channels, win_len), device=audio_stream.device, dtype=fft_dtype) for _ in range(2)]

        out_shape = (
            (B, 2, padded_len) if (not stereo_input or not self.stereo_loss) else (B, 2, 2, padded_len)
        )
        out = torch.zeros(out_shape, device=audio_stream.device, dtype=fft_dtype)
        out_pos = 0

        for t in range(0, padded_len, emit_len):
            chunk = audio_stream[..., t : t + emit_len]
            if chunk.numel() == 0:
                continue
            stft_buf = torch.cat([stft_buf, chunk], dim=-1)

            new_frames = []
            while stft_buf.size(-1) >= win_len:
                frame_td = stft_buf[..., :win_len]
                stft_buf = stft_buf[..., hop:]
                frame_td_fft = frame_td.to(dtype=fft_dtype)
                frame_win = frame_td_fft * window
                frame_cpx = torch.fft.rfft(frame_win, n=self.n_fft, dim=-1)
                new_frames.append(frame_cpx)

            if not new_frames:
                continue

            new_frames_cpx = torch.stack(new_frames, dim=-1)  # (B, C, F, Tnew)
            num_new_frames = new_frames_cpx.size(-1)

            if stereo_input:
                mix_cpx_buf = new_frames_cpx if mix_cpx_buf is None else torch.cat([mix_cpx_buf, new_frames_cpx], dim=-1)
                if mix_cpx_buf.size(-1) > max_frames:
                    mix_cpx_buf = mix_cpx_buf[..., -max_frames:]
                mix_twoch = _complex_to_twoch_multi(mix_cpx_buf)
                mix_cpx_crop = mix_cpx_buf[:, :, :-1, :]
            else:
                new_frames_mono = new_frames_cpx[:, 0]
                mix_cpx_buf = new_frames_mono if mix_cpx_buf is None else torch.cat([mix_cpx_buf, new_frames_mono], dim=-1)
                if mix_cpx_buf.size(-1) > max_frames:
                    mix_cpx_buf = mix_cpx_buf[..., -max_frames:]
                mix_twoch = _complex_to_twoch(mix_cpx_buf)
                mix_cpx_crop = mix_cpx_buf[:, :-1, :]

            if mix_twoch.dtype != torch.float32:
                mix_twoch = mix_twoch.float()
            masks = self.mask_net(mix_twoch, vec_feature, activation=self.activation, return_tuple=False)

            target_F, target_T = mix_cpx_crop.shape[-2], mix_cpx_crop.shape[-1]
            if (masks.shape[3], masks.shape[4]) != (target_F, target_T):
                Bb, S, Cc, Fm, Tm = masks.shape
                masks_4d = masks.reshape(Bb * S, Cc, Fm, Tm)
                masks_4d = F.interpolate(masks_4d, size=(target_F, target_T), mode="nearest")
                masks = masks_4d.reshape(Bb, S, Cc, target_F, target_T)

            mix_cpx_crop_new = mix_cpx_crop[..., -num_new_frames:]
            masks_new = masks[..., -num_new_frames:]

            src_specs = []
            for src_idx in range(2):
                mask_src = masks_new[:, src_idx]
                if stereo_input:
                    if mask_src.size(1) != 4:
                        raise ValueError(
                            f"stereo mask requires output_nc=4 (L_real,L_imag,R_real,R_imag), got {mask_src.size(1)}"
                        )
                    mask_L = mask_src[:, 0:2]
                    mask_R = mask_src[:, 2:4]
                    mask_L_cpx = _twoch_to_complex(mask_L)
                    mask_R_cpx = _twoch_to_complex(mask_R)
                    src_L = mix_cpx_crop_new[:, 0] * mask_L_cpx
                    src_R = mix_cpx_crop_new[:, 1] * mask_R_cpx
                    src_cpx_crop = torch.stack([src_L, src_R], dim=1)
                else:
                    mask_cpx = _twoch_to_complex(mask_src)
                    src_cpx_crop = (mix_cpx_crop_new * mask_cpx).unsqueeze(1)
                src_cpx = F.pad(src_cpx_crop, (0, 0, 0, 1))
                src_specs.append(src_cpx)

            for frame_idx in range(num_new_frames):
                out_hops = []
                for src_idx in range(2):
                    spec_frame = src_specs[src_idx][..., frame_idx]
                    td = torch.fft.irfft(spec_frame, n=self.n_fft, dim=-1)
                    td = td * window
                    ola_num[src_idx] = ola_num[src_idx] + td
                    ola_den[src_idx] = ola_den[src_idx] + window_sq
                    denom = ola_den[src_idx][:, :, :hop].clamp_min(1e-8)
                    out_hop = ola_num[src_idx][:, :, :hop] / denom
                    out_hops.append(out_hop)
                    zeros = torch.zeros_like(ola_num[src_idx][:, :, :hop])
                    ola_num[src_idx] = torch.cat([ola_num[src_idx][:, :, hop:], zeros], dim=-1)
                    ola_den[src_idx] = torch.cat([ola_den[src_idx][:, :, hop:], zeros], dim=-1)

                if stereo_input and self.stereo_loss:
                    out[:, 0, :, out_pos : out_pos + hop] = out_hops[0]
                    out[:, 1, :, out_pos : out_pos + hop] = out_hops[1]
                elif stereo_input and not self.stereo_loss:
                    out[:, 0, out_pos : out_pos + hop] = out_hops[0].mean(dim=1)
                    out[:, 1, out_pos : out_pos + hop] = out_hops[1].mean(dim=1)
                else:
                    out[:, 0, out_pos : out_pos + hop] = out_hops[0][:, 0]
                    out[:, 1, out_pos : out_pos + hop] = out_hops[1][:, 0]
                out_pos += hop

        tail_len = max(0, win_len - hop)
        if tail_len > 0:
            for src_idx in range(2):
                denom = ola_den[src_idx][:, :, :tail_len].clamp_min(1e-8)
                tail = ola_num[src_idx][:, :, :tail_len] / denom
                if stereo_input and self.stereo_loss:
                    out[:, src_idx, :, out_pos : out_pos + tail_len] = tail
                elif stereo_input and not self.stereo_loss:
                    out[:, src_idx, out_pos : out_pos + tail_len] = tail.mean(dim=1)
                else:
                    out[:, src_idx, out_pos : out_pos + tail_len] = tail[:, 0]
            out_pos += tail_len

        return out[..., :Tlen]

    def forward(self, audio_mix, vec_feature, return_masks=False):
        if self.streaming_train:
            return self._forward_streaming(audio_mix, vec_feature)
        return self._forward_full(audio_mix, vec_feature, return_masks=return_masks)

