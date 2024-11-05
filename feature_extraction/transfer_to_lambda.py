make
CC	libavdevice/alldevices.o
CC	libavdevice/avdevice.o
libavdevice/avdevice.c: In function ‘device_next’:
libavdevice/avdevice.c:88:13: warning: ‘av_oformat_next’ is deprecated [-Wdeprecated-declarations]
   88 |             if (!(prev = av_oformat_next(prev)))
      |             ^~
In file included from libavdevice/avdevice.h:51,
                 from libavdevice/avdevice.c:23:
./libavformat/avformat.h:2095:17: note: declared here
 2095 | AVOutputFormat *av_oformat_next(const AVOutputFormat *f);
      |                 ^~~~~~~~~~~~~~~
libavdevice/avdevice.c:92:13: warning: ‘av_iformat_next’ is deprecated [-Wdeprecated-declarations]
   92 |             if (!(prev = av_iformat_next(prev)))
      |             ^~
./libavformat/avformat.h:2087:17: note: declared here
 2087 | AVInputFormat  *av_iformat_next(const AVInputFormat  *f);
      |                 ^~~~~~~~~~~~~~~
CC	libavdevice/fbdev_common.o
CC	libavdevice/fbdev_dec.o
CC	libavdevice/fbdev_enc.o
CC	libavdevice/lavfi.o
CC	libavdevice/oss.o
CC	libavdevice/oss_dec.o
CC	libavdevice/oss_enc.o
CC	libavdevice/timefilter.o
CC	libavdevice/utils.o
CC	libavdevice/v4l2-common.o
CC	libavdevice/v4l2.o
libavdevice/v4l2.c: In function ‘v4l2_get_device_list’:
libavdevice/v4l2.c:1054:58: warning: ‘%s’ directive output may be truncated writing up to 255 bytes into a region of size 251 [-Wformat-truncation=]
 1054 |         snprintf(device_name, sizeof(device_name), "/dev/%s", entry->d_name);
      |                                                          ^~
In file included from /usr/include/stdio.h:980,
                 from ./libavformat/avformat.h:311,
                 from ./libavformat/internal.h:27,
                 from libavdevice/v4l2-common.h:24,
                 from libavdevice/v4l2.c:35:
In function ‘snprintf’,
    inlined from ‘v4l2_get_device_list’ at libavdevice/v4l2.c:1054:9:
/usr/include/x86_64-linux-gnu/bits/stdio2.h:54:10: note: ‘__builtin___snprintf_chk’ output between 6 and 261 bytes into a destination of size 256
   54 |   return __builtin___snprintf_chk (__s, __n, __USE_FORTIFY_LEVEL - 1,
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   55 |                                    __glibc_objsize (__s), __fmt,
      |                                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   56 |                                    __va_arg_pack ());
      |                                    ~~~~~~~~~~~~~~~~~
CC	libavdevice/v4l2enc.o
CC	libavdevice/xcbgrab.o
AR	libavdevice/libavdevice.a
CC	libavfilter/aeval.o
CC	libavfilter/af_acontrast.o
CC	libavfilter/af_acopy.o
CC	libavfilter/af_acrossover.o
CC	libavfilter/af_acrusher.o
CC	libavfilter/af_adeclick.o
CC	libavfilter/af_adelay.o
CC	libavfilter/af_aderivative.o
CC	libavfilter/af_aecho.o
CC	libavfilter/af_aemphasis.o
CC	libavfilter/af_afade.o
CC	libavfilter/af_afftdn.o
CC	libavfilter/af_afftfilt.o
CC	libavfilter/af_afir.o
CC	libavfilter/af_aformat.o
CC	libavfilter/af_agate.o
CC	libavfilter/af_aiir.o
CC	libavfilter/af_alimiter.o
CC	libavfilter/af_amerge.o
CC	libavfilter/af_amix.o
CC	libavfilter/af_amultiply.o
CC	libavfilter/af_anequalizer.o
CC	libavfilter/af_anlmdn.o
CC	libavfilter/af_anlms.o
CC	libavfilter/af_anull.o
CC	libavfilter/af_apad.o
CC	libavfilter/af_aphaser.o
CC	libavfilter/af_apulsator.o
CC	libavfilter/af_aresample.o
CC	libavfilter/af_arnndn.o
CC	libavfilter/af_asetnsamples.o
CC	libavfilter/af_asetrate.o
CC	libavfilter/af_ashowinfo.o
CC	libavfilter/af_asoftclip.o
CC	libavfilter/af_astats.o
CC	libavfilter/af_asubboost.o
CC	libavfilter/af_atempo.o
CC	libavfilter/af_axcorrelate.o
CC	libavfilter/af_biquads.o
CC	libavfilter/af_channelmap.o
CC	libavfilter/af_channelsplit.o
CC	libavfilter/af_chorus.o
CC	libavfilter/af_compand.o
CC	libavfilter/af_compensationdelay.o
CC	libavfilter/af_crossfeed.o
CC	libavfilter/af_crystalizer.o
CC	libavfilter/af_dcshift.o
CC	libavfilter/af_deesser.o
CC	libavfilter/af_drmeter.o
CC	libavfilter/af_dynaudnorm.o
CC	libavfilter/af_earwax.o
CC	libavfilter/af_extrastereo.o
CC	libavfilter/af_firequalizer.o
CC	libavfilter/af_flanger.o
CC	libavfilter/af_haas.o
CC	libavfilter/af_hdcd.o
CC	libavfilter/af_headphone.o
CC	libavfilter/af_join.o
CC	libavfilter/af_loudnorm.o
CC	libavfilter/af_mcompand.o
CC	libavfilter/af_pan.o
CC	libavfilter/af_replaygain.o
CC	libavfilter/af_sidechaincompress.o
CC	libavfilter/af_silencedetect.o
CC	libavfilter/af_silenceremove.o
CC	libavfilter/af_stereotools.o
CC	libavfilter/af_stereowiden.o
CC	libavfilter/af_superequalizer.o
CC	libavfilter/af_surround.o
CC	libavfilter/af_tremolo.o
CC	libavfilter/af_vibrato.o
CC	libavfilter/af_volume.o
CC	libavfilter/af_volumedetect.o
CC	libavfilter/allfilters.o
CC	libavfilter/asink_anullsink.o
CC	libavfilter/asrc_afirsrc.o
CC	libavfilter/asrc_anoisesrc.o
CC	libavfilter/asrc_anullsrc.o
CC	libavfilter/asrc_hilbert.o
CC	libavfilter/asrc_sinc.o
CC	libavfilter/asrc_sine.o
CC	libavfilter/audio.o
CC	libavfilter/avf_abitscope.o
CC	libavfilter/avf_ahistogram.o
CC	libavfilter/avf_aphasemeter.o
CC	libavfilter/avf_avectorscope.o
CC	libavfilter/avf_concat.o
CC	libavfilter/avf_showcqt.o
CC	libavfilter/avf_showfreqs.o
CC	libavfilter/avf_showspatial.o
CC	libavfilter/avf_showspectrum.o
CC	libavfilter/avf_showvolume.o
CC	libavfilter/avf_showwaves.o
CC	libavfilter/avfilter.o
CC	libavfilter/avfiltergraph.o
libavfilter/avfiltergraph.c: In function ‘avfilter_graph_free’:
libavfilter/avfiltergraph.c:135:5: warning: ‘resample_lavr_opts’ is deprecated [-Wdeprecated-declarations]
  135 |     av_freep(&(*graph)->resample_lavr_opts);
      |     ^~~~~~~~
In file included from libavfilter/avfiltergraph.c:39:
libavfilter/avfilter.h:847:32: note: declared here
  847 |     attribute_deprecated char *resample_lavr_opts;   ///< libavresample options to use for the auto-inserted resample filters
      |                                ^~~~~~~~~~~~~~~~~~
CC	libavfilter/boxblur.o
CC	libavfilter/buffersink.o
CC	libavfilter/buffersrc.o
CC	libavfilter/colorspace.o
CC	libavfilter/colorspacedsp.o
CC	libavfilter/dnn/dnn_backend_native.o
CC	libavfilter/dnn/dnn_backend_native_layer_conv2d.o
CC	libavfilter/dnn/dnn_backend_native_layer_depth2space.o
CC	libavfilter/dnn/dnn_backend_native_layer_mathbinary.o
CC	libavfilter/dnn/dnn_backend_native_layer_mathunary.o
CC	libavfilter/dnn/dnn_backend_native_layer_maximum.o
CC	libavfilter/dnn/dnn_backend_native_layer_pad.o
CC	libavfilter/dnn/dnn_backend_native_layers.o
CC	libavfilter/dnn/dnn_interface.o
CC	libavfilter/drawutils.o
CC	libavfilter/ebur128.o
CC	libavfilter/f_bench.o
CC	libavfilter/f_cue.o
CC	libavfilter/f_drawgraph.o
CC	libavfilter/f_ebur128.o
CC	libavfilter/f_graphmonitor.o
CC	libavfilter/f_interleave.o
CC	libavfilter/f_loop.o
CC	libavfilter/f_metadata.o
CC	libavfilter/f_perms.o
CC	libavfilter/f_realtime.o
CC	libavfilter/f_reverse.o
CC	libavfilter/f_select.o
CC	libavfilter/f_sendcmd.o
CC	libavfilter/f_sidedata.o
CC	libavfilter/f_streamselect.o
CC	libavfilter/fifo.o
CC	libavfilter/formats.o
CC	libavfilter/framepool.o
CC	libavfilter/framequeue.o
CC	libavfilter/framesync.o
CC	libavfilter/generate_wave_table.o
CC	libavfilter/graphdump.o
CC	libavfilter/graphparser.o
CC	libavfilter/lavfutils.o
libavfilter/lavfutils.c: In function ‘ff_load_image’:
libavfilter/lavfutils.c:91:5: warning: ‘avcodec_decode_video2’ is deprecated [-Wdeprecated-declarations]
   91 |     ret = avcodec_decode_video2(codec_ctx, frame, &frame_decoded, &pkt);
      |     ^~~
In file included from ./libavformat/avformat.h:312,
                 from libavfilter/lavfutils.h:27,
                 from libavfilter/lavfutils.c:22:
./libavcodec/avcodec.h:3073:5: note: declared here
 3073 | int avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,
      |     ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/lswsutils.o
CC	libavfilter/motion_estimation.o
CC	libavfilter/pthread.o
CC	libavfilter/scale_eval.o
CC	libavfilter/scene_sad.o
CC	libavfilter/setpts.o
CC	libavfilter/settb.o
CC	libavfilter/split.o
CC	libavfilter/src_movie.o
libavfilter/src_movie.c: In function ‘open_stream’:
libavfilter/src_movie.c:175:5: warning: ‘refcounted_frames’ is deprecated [-Wdeprecated-declarations]
  175 |     st->codec_ctx->refcounted_frames = 1;
      |     ^~
In file included from libavfilter/src_movie.c:41:
./libavcodec/avcodec.h:1357:9: note: declared here
 1357 |     int refcounted_frames;
      |         ^~~~~~~~~~~~~~~~~
libavfilter/src_movie.c: In function ‘movie_push_frame’:
libavfilter/src_movie.c:530:9: warning: ‘avcodec_decode_video2’ is deprecated [-Wdeprecated-declarations]
  530 |         ret = avcodec_decode_video2(st->codec_ctx, frame, &got_frame, pkt);
      |         ^~~
./libavcodec/avcodec.h:3073:5: note: declared here
 3073 | int avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,
      |     ^~~~~~~~~~~~~~~~~~~~~
libavfilter/src_movie.c:533:9: warning: ‘avcodec_decode_audio4’ is deprecated [-Wdeprecated-declarations]
  533 |         ret = avcodec_decode_audio4(st->codec_ctx, frame, &got_frame, pkt);
      |         ^~~
./libavcodec/avcodec.h:3024:5: note: declared here
 3024 | int avcodec_decode_audio4(AVCodecContext *avctx, AVFrame *frame,
      |     ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/transform.o
CC	libavfilter/trim.o
CC	libavfilter/vaf_spectrumsynth.o
CC	libavfilter/vf_addroi.o
CC	libavfilter/vf_alphamerge.o
CC	libavfilter/vf_amplify.o
CC	libavfilter/vf_aspect.o
CC	libavfilter/vf_atadenoise.o
CC	libavfilter/vf_avgblur.o
CC	libavfilter/vf_bbox.o
CC	libavfilter/vf_bilateral.o
CC	libavfilter/vf_bitplanenoise.o
CC	libavfilter/vf_blackdetect.o
CC	libavfilter/vf_blackframe.o
CC	libavfilter/vf_blend.o
CC	libavfilter/vf_bm3d.o
CC	libavfilter/vf_boxblur.o
CC	libavfilter/vf_bwdif.o
CC	libavfilter/vf_cas.o
CC	libavfilter/vf_chromakey.o
CC	libavfilter/vf_chromashift.o
CC	libavfilter/vf_ciescope.o
CC	libavfilter/vf_codecview.o
libavfilter/vf_codecview.c: In function ‘filter_frame’:
libavfilter/vf_codecview.c:223:9: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  223 |         int8_t *qp_table = av_frame_get_qp_table(frame, &qstride, &qp_type);
      |         ^~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/vf_codecview.c:35:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_colorbalance.o
CC	libavfilter/vf_colorchannelmixer.o
CC	libavfilter/vf_colorconstancy.o
CC	libavfilter/vf_colorkey.o
CC	libavfilter/vf_colorlevels.o
CC	libavfilter/vf_colormatrix.o
CC	libavfilter/vf_colorspace.o
CC	libavfilter/vf_convolution.o
CC	libavfilter/vf_convolve.o
CC	libavfilter/vf_copy.o
CC	libavfilter/vf_cover_rect.o
CC	libavfilter/vf_crop.o
CC	libavfilter/vf_cropdetect.o
CC	libavfilter/vf_curves.o
CC	libavfilter/vf_datascope.o
CC	libavfilter/vf_dblur.o
CC	libavfilter/vf_dctdnoiz.o
CC	libavfilter/vf_deband.o
CC	libavfilter/vf_deblock.o
CC	libavfilter/vf_decimate.o
CC	libavfilter/vf_dedot.o
CC	libavfilter/vf_deflicker.o
CC	libavfilter/vf_dejudder.o
CC	libavfilter/vf_delogo.o
CC	libavfilter/vf_derain.o
CC	libavfilter/vf_deshake.o
CC	libavfilter/vf_despill.o
CC	libavfilter/vf_detelecine.o
CC	libavfilter/vf_displace.o
CC	libavfilter/vf_dnn_processing.o
CC	libavfilter/vf_drawbox.o
CC	libavfilter/vf_edgedetect.o
CC	libavfilter/vf_elbg.o
CC	libavfilter/vf_entropy.o
CC	libavfilter/vf_eq.o
CC	libavfilter/vf_extractplanes.o
CC	libavfilter/vf_fade.o
CC	libavfilter/vf_fftdnoiz.o
CC	libavfilter/vf_fftfilt.o
CC	libavfilter/vf_field.o
CC	libavfilter/vf_fieldhint.o
CC	libavfilter/vf_fieldmatch.o
CC	libavfilter/vf_fieldorder.o
CC	libavfilter/vf_fillborders.o
CC	libavfilter/vf_find_rect.o
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:79: warning: array subscript 7 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 | score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                            ~~~~~~~~~~~~~~~~~^~~~~~

libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:41:14: note: while referencing ‘needle_frame’
   41 |     AVFrame *needle_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:54: warning: array subscript 7 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 |             float score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                   ~~~~~~~~~~~~~~~~~~~^~~~~~
libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:42:14: note: while referencing ‘haystack_frame’
   42 |     AVFrame *haystack_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~~~
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:79: warning: array subscript 6 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 | score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                            ~~~~~~~~~~~~~~~~~^~~~~~

libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:41:14: note: while referencing ‘needle_frame’
   41 |     AVFrame *needle_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:54: warning: array subscript 6 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 |             float score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                   ~~~~~~~~~~~~~~~~~~~^~~~~~
libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:42:14: note: while referencing ‘haystack_frame’
   42 |     AVFrame *haystack_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~~~
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:79: warning: array subscript 5 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 | score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                            ~~~~~~~~~~~~~~~~~^~~~~~

libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:41:14: note: while referencing ‘needle_frame’
   41 |     AVFrame *needle_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~
In function ‘search’,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search’ at libavfilter/vf_find_rect.c:161:9,
    inlined from ‘search.constprop’ at libavfilter/vf_find_rect.c:161:9:
libavfilter/vf_find_rect.c:170:54: warning: array subscript 5 is above array bounds of ‘AVFrame *[5]’ [-Warray-bounds=]
  170 |             float score = compare(foc->haystack_frame[pass], foc->needle_frame[pass], x, y);
      |                                   ~~~~~~~~~~~~~~~~~~~^~~~~~
libavfilter/vf_find_rect.c: In function ‘search.constprop’:
libavfilter/vf_find_rect.c:42:14: note: while referencing ‘haystack_frame’
   42 |     AVFrame *haystack_frame[MAX_MIPMAPS];
      |              ^~~~~~~~~~~~~~
CC	libavfilter/vf_floodfill.o
CC	libavfilter/vf_format.o
CC	libavfilter/vf_fps.o
CC	libavfilter/vf_framepack.o
CC	libavfilter/vf_framerate.o
CC	libavfilter/vf_framestep.o
CC	libavfilter/vf_freezedetect.o
CC	libavfilter/vf_freezeframes.o
CC	libavfilter/vf_fspp.o
libavfilter/vf_fspp.c: In function ‘filter_frame’:
libavfilter/vf_fspp.c:585:9: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  585 |         qp_table = av_frame_get_qp_table(in, &qp_stride, &fspp->qscale_type);
      |         ^~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/internal.h:28,
                 from libavfilter/vf_fspp.c:42:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_gblur.o
CC	libavfilter/vf_geq.o
libavfilter/vf_geq.c: In function ‘geq_init’:
libavfilter/vf_geq.c:254:51: warning: ‘%d’ directive output may be truncated writing between 1 and 11 bytes into a region of size 8 [-Wformat-truncation=]
  254 |         snprintf(bps_string, sizeof(bps_string), "%d", (1<<geq->bps) - 1);
      |                                                   ^~
libavfilter/vf_geq.c:254:50: note: directive argument in the range [-2147483648, 2147483646]
  254 |         snprintf(bps_string, sizeof(bps_string), "%d", (1<<geq->bps) - 1);
      |                                                  ^~~~
In file included from /usr/include/stdio.h:980,
                 from ./libavutil/common.h:38,
                 from ./libavutil/avutil.h:296,
                 from ./libavutil/avassert.h:31,
                 from libavfilter/vf_geq.c:29:
In function ‘snprintf’,
    inlined from ‘geq_init’ at libavfilter/vf_geq.c:254:9:
/usr/include/x86_64-linux-gnu/bits/stdio2.h:54:10: note: ‘__builtin___snprintf_chk’ output between 2 and 12 bytes into a destination of size 8
   54 |   return __builtin___snprintf_chk (__s, __n, __USE_FORTIFY_LEVEL - 1,
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   55 |                                    __glibc_objsize (__s), __fmt,
      |                                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   56 |                                    __va_arg_pack ());
      |                                    ~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_gradfun.o
CC	libavfilter/vf_hflip.o
CC	libavfilter/vf_histeq.o
CC	libavfilter/vf_histogram.o
CC	libavfilter/vf_hqdn3d.o
CC	libavfilter/vf_hqx.o
CC	libavfilter/vf_hue.o
CC	libavfilter/vf_hwdownload.o
CC	libavfilter/vf_hwmap.o
CC	libavfilter/vf_hwupload.o
CC	libavfilter/vf_hysteresis.o
CC	libavfilter/vf_idet.o
CC	libavfilter/vf_il.o
CC	libavfilter/vf_kerndeint.o
CC	libavfilter/vf_lagfun.o
CC	libavfilter/vf_lenscorrection.o
CC	libavfilter/vf_limiter.o
CC	libavfilter/vf_lumakey.o
CC	libavfilter/vf_lut.o
CC	libavfilter/vf_lut2.o
CC	libavfilter/vf_lut3d.o
CC	libavfilter/vf_maskedclamp.o
CC	libavfilter/vf_maskedmerge.o
CC	libavfilter/vf_maskedminmax.o
CC	libavfilter/vf_maskedthreshold.o
CC	libavfilter/vf_maskfun.o
CC	libavfilter/vf_mcdeint.o
libavfilter/vf_mcdeint.c: In function ‘filter_frame’:
libavfilter/vf_mcdeint.c:189:5: warning: ‘avcodec_encode_video2’ is deprecated [-Wdeprecated-declarations]
  189 |     ret = avcodec_encode_video2(mcdeint->enc_ctx, &pkt, inpic, &got_frame);
      |     ^~~
In file included from libavfilter/vf_mcdeint.c:54:
./libavcodec/avcodec.h:3691:5: note: declared here
 3691 | int avcodec_encode_video2(AVCodecContext *avctx, AVPacket *avpkt,
      |     ^~~~~~~~~~~~~~~~~~~~~
libavfilter/vf_mcdeint.c:193:5: warning: ‘coded_frame’ is deprecated [-Wdeprecated-declarations]
  193 |     frame_dec = mcdeint->enc_ctx->coded_frame;
      |     ^~~~~~~~~
./libavcodec/avcodec.h:1776:35: note: declared here
 1776 |     attribute_deprecated AVFrame *coded_frame;
      |                                   ^~~~~~~~~~~
CC	libavfilter/vf_median.o
CC	libavfilter/vf_mergeplanes.o
CC	libavfilter/vf_mestimate.o
CC	libavfilter/vf_midequalizer.o
CC	libavfilter/vf_minterpolate.o
CC	libavfilter/vf_mix.o
CC	libavfilter/vf_mpdecimate.o
CC	libavfilter/vf_neighbor.o
CC	libavfilter/vf_nlmeans.o
CC	libavfilter/vf_nnedi.o
CC	libavfilter/vf_noise.o
CC	libavfilter/vf_normalize.o
CC	libavfilter/vf_null.o
CC	libavfilter/vf_overlay.o
CC	libavfilter/vf_owdenoise.o
CC	libavfilter/vf_pad.o
CC	libavfilter/vf_palettegen.o
CC	libavfilter/vf_paletteuse.o
CC	libavfilter/vf_perspective.o
CC	libavfilter/vf_phase.o
CC	libavfilter/vf_photosensitivity.o
CC	libavfilter/vf_pixdesctest.o
CC	libavfilter/vf_pp.o
libavfilter/vf_pp.c: In function ‘pp_filter_frame’:
libavfilter/vf_pp.c:140:5: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  140 |     qp_table = av_frame_get_qp_table(inbuf, &qstride, &qp_type);
      |     ^~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/internal.h:28,
                 from libavfilter/vf_pp.c:29:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_pp7.o
libavfilter/vf_pp7.c: In function ‘filter_frame’:
libavfilter/vf_pp7.c:328:9: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  328 |         qp_table = av_frame_get_qp_table(in, &qp_stride, &pp7->qscale_type);
      |         ^~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/internal.h:28,
                 from libavfilter/vf_pp7.c:34:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_premultiply.o
CC	libavfilter/vf_pseudocolor.o
CC	libavfilter/vf_psnr.o
CC	libavfilter/vf_pullup.o
CC	libavfilter/vf_qp.o
libavfilter/vf_qp.c: In function ‘filter_frame’:
libavfilter/vf_qp.c:113:5: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  113 |     in_qp_table = av_frame_get_qp_table(in, &stride, &type);
      |     ^~~~~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/vf_qp.c:26:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
libavfilter/vf_qp.c:114:5: warning: ‘av_frame_set_qp_table’ is deprecated [-Wdeprecated-declarations]
  114 |     av_frame_set_qp_table(out, out_qp_table_buf, s->qstride, type);
      |     ^~~~~~~~~~~~~~~~~~~~~
./libavutil/frame.h:727:5: note: declared here
  727 | int av_frame_set_qp_table(AVFrame *f, AVBufferRef *buf, int stride, int type);
      |     ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_random.o
CC	libavfilter/vf_readeia608.o
CC	libavfilter/vf_readvitc.o
CC	libavfilter/vf_remap.o
CC	libavfilter/vf_removegrain.o
CC	libavfilter/vf_removelogo.o
CC	libavfilter/vf_repeatfields.o
CC	libavfilter/vf_rotate.o
CC	libavfilter/vf_sab.o
CC	libavfilter/vf_scale.o
CC	libavfilter/vf_scdet.o
CC	libavfilter/vf_scroll.o
CC	libavfilter/vf_selectivecolor.o
CC	libavfilter/vf_separatefields.o
CC	libavfilter/vf_setparams.o
CC	libavfilter/vf_showinfo.o
CC	libavfilter/vf_showpalette.o
CC	libavfilter/vf_shuffleframes.o
CC	libavfilter/vf_shuffleplanes.o
CC	libavfilter/vf_signalstats.o
CC	libavfilter/vf_signature.o
CC	libavfilter/vf_smartblur.o
CC	libavfilter/vf_spp.o
libavfilter/vf_spp.c: In function ‘filter_frame’:
libavfilter/vf_spp.c:374:9: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  374 |         qp_table = av_frame_get_qp_table(in, &qp_stride, &s->qscale_type);
      |         ^~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/internal.h:28,
                 from libavfilter/vf_spp.c:38:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_sr.o
CC	libavfilter/vf_ssim.o
CC	libavfilter/vf_stack.o
CC	libavfilter/vf_stereo3d.o
CC	libavfilter/vf_super2xsai.o
CC	libavfilter/vf_swaprect.o
CC	libavfilter/vf_swapuv.o
CC	libavfilter/vf_telecine.o
CC	libavfilter/vf_threshold.o
CC	libavfilter/vf_thumbnail.o
CC	libavfilter/vf_tile.o
CC	libavfilter/vf_tinterlace.o
CC	libavfilter/vf_tonemap.o
CC	libavfilter/vf_tpad.o
CC	libavfilter/vf_transpose.o
CC	libavfilter/vf_unsharp.o
CC	libavfilter/vf_untile.o
CC	libavfilter/vf_uspp.o
libavfilter/vf_uspp.c: In function ‘filter’:
libavfilter/vf_uspp.c:253:9: warning: ‘avcodec_encode_video2’ is deprecated [-Wdeprecated-declarations]
  253 |         ret = avcodec_encode_video2(p->avctx_enc[i], &pkt, p->frame, &got_pkt_ptr);
      |         ^~~
In file included from libavfilter/internal.h:35,
                 from libavfilter/vf_uspp.c:34:
./libavcodec/avcodec.h:3691:5: note: declared here
 3691 | int avcodec_encode_video2(AVCodecContext *avctx, AVPacket *avpkt,
      |     ^~~~~~~~~~~~~~~~~~~~~
libavfilter/vf_uspp.c:259:9: warning: ‘coded_frame’ is deprecated [-Wdeprecated-declarations]
  259 |         p->frame_dec = p->avctx_enc[i]->coded_frame;
      |         ^
./libavcodec/avcodec.h:1776:35: note: declared here
 1776 |     attribute_deprecated AVFrame *coded_frame;
      |                                   ^~~~~~~~~~~
libavfilter/vf_uspp.c: In function ‘filter_frame’:
libavfilter/vf_uspp.c:395:9: warning: ‘av_frame_get_qp_table’ is deprecated [-Wdeprecated-declarations]
  395 |         qp_table = av_frame_get_qp_table(in, &qp_stride, &uspp->qscale_type);
      |         ^~~~~~~~
In file included from libavfilter/avfilter.h:44,
                 from libavfilter/internal.h:28:
./libavutil/frame.h:725:9: note: declared here
  725 | int8_t *av_frame_get_qp_table(AVFrame *f, int *stride, int *type);
      |         ^~~~~~~~~~~~~~~~~~~~~
CC	libavfilter/vf_v360.o
CC	libavfilter/vf_vaguedenoiser.o
CC	libavfilter/vf_vectorscope.o
CC	libavfilter/vf_vflip.o
CC	libavfilter/vf_vfrdet.o
CC	libavfilter/vf_vibrance.o
CC	libavfilter/vf_vignette.o
CC	libavfilter/vf_vmafmotion.o
CC	libavfilter/vf_w3fdif.o
CC	libavfilter/vf_waveform.o
CC	libavfilter/vf_weave.o
CC	libavfilter/vf_xbr.o
CC	libavfilter/vf_xfade.o
CC	libavfilter/vf_xmedian.o
CC	libavfilter/vf_yadif.o
CC	libavfilter/vf_yaepblur.o
CC	libavfilter/vf_zoompan.o
CC	libavfilter/video.o
CC	libavfilter/vsink_nullsink.o
CC	libavfilter/vsrc_cellauto.o
CC	libavfilter/vsrc_gradients.o
CC	libavfilter/vsrc_life.o
CC	libavfilter/vsrc_mandelbrot.o
CC	libavfilter/vsrc_mptestsrc.o
CC	libavfilter/vsrc_sierpinski.o
CC	libavfilter/vsrc_testsrc.o
X86ASM	libavfilter/x86/af_afir.o
STRIP	libavfilter/x86/af_afir.o
CC	libavfilter/x86/af_afir_init.o
X86ASM	libavfilter/x86/af_anlmdn.o
STRIP	libavfilter/x86/af_anlmdn.o
CC	libavfilter/x86/af_anlmdn_init.o
X86ASM	libavfilter/x86/af_volume.o
STRIP	libavfilter/x86/af_volume.o
CC	libavfilter/x86/af_volume_init.o
X86ASM	libavfilter/x86/avf_showcqt.o
STRIP	libavfilter/x86/avf_showcqt.o
CC	libavfilter/x86/avf_showcqt_init.o
X86ASM	libavfilter/x86/colorspacedsp.o
STRIP	libavfilter/x86/colorspacedsp.o
CC	libavfilter/x86/colorspacedsp_init.o
X86ASM	libavfilter/x86/scene_sad.o
STRIP	libavfilter/x86/scene_sad.o
CC	libavfilter/x86/scene_sad_init.o
X86ASM	libavfilter/x86/vf_atadenoise.o
STRIP	libavfilter/x86/vf_atadenoise.o
CC	libavfilter/x86/vf_atadenoise_init.o
X86ASM	libavfilter/x86/vf_blend.o
libavfilter/x86/vf_blend.asm:416: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:417: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:418: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:419: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:420: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:421: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:422: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:423: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:86: ... from macro `GRAINEXTRACT' defined here
libavfilter/x86/vf_blend.asm:426: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:186: ... from macro `AVERAGE' defined here
libavfilter/x86/vf_blend.asm:427: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:207: ... from macro `GRAINMERGE' defined here
libavfilter/x86/vf_blend.asm:429: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:293: ... from macro `PHOENIX' defined here
libavfilter/x86/vf_blend.asm:430: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:315: ... from macro `DIFFERENCE' defined here
libavfilter/x86/vf_blend.asm:432: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:344: ... from macro `EXTREMITY' defined here
libavfilter/x86/vf_blend.asm:433: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:379: ... from macro `NEGATION' defined here
libavfilter/x86/vf_blend.asm:445: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:315: ... from macro `DIFFERENCE' defined here
libavfilter/x86/vf_blend.asm:446: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:344: ... from macro `EXTREMITY' defined here
libavfilter/x86/vf_blend.asm:447: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:379: ... from macro `NEGATION' defined here
libavfilter/x86/vf_blend.asm:463: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:464: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:465: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:466: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:467: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:468: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:469: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:70: ... from macro `BLEND_SIMPLE' defined here
libavfilter/x86/vf_blend.asm:470: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:86: ... from macro `GRAINEXTRACT' defined here
libavfilter/x86/vf_blend.asm:473: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:186: ... from macro `AVERAGE' defined here
libavfilter/x86/vf_blend.asm:474: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:207: ... from macro `GRAINMERGE' defined here
libavfilter/x86/vf_blend.asm:476: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:293: ... from macro `PHOENIX' defined here
libavfilter/x86/vf_blend.asm:478: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:315: ... from macro `DIFFERENCE' defined here
libavfilter/x86/vf_blend.asm:479: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:344: ... from macro `EXTREMITY' defined here
libavfilter/x86/vf_blend.asm:480: warning: dropping trailing empty parameter in call to multi-line macro `BLEND_INIT' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_blend.asm:379: ... from macro `NEGATION' defined here
STRIP	libavfilter/x86/vf_blend.o
CC	libavfilter/x86/vf_blend_init.o
X86ASM	libavfilter/x86/vf_bwdif.o
STRIP	libavfilter/x86/vf_bwdif.o
CC	libavfilter/x86/vf_bwdif_init.o
X86ASM	libavfilter/x86/vf_convolution.o
STRIP	libavfilter/x86/vf_convolution.o
CC	libavfilter/x86/vf_convolution_init.o
X86ASM	libavfilter/x86/vf_eq.o
STRIP	libavfilter/x86/vf_eq.o
CC	libavfilter/x86/vf_eq_init.o
X86ASM	libavfilter/x86/vf_framerate.o
STRIP	libavfilter/x86/vf_framerate.o
CC	libavfilter/x86/vf_framerate_init.o
X86ASM	libavfilter/x86/vf_fspp.o
STRIP	libavfilter/x86/vf_fspp.o
CC	libavfilter/x86/vf_fspp_init.o
X86ASM	libavfilter/x86/vf_gblur.o
STRIP	libavfilter/x86/vf_gblur.o
CC	libavfilter/x86/vf_gblur_init.o
X86ASM	libavfilter/x86/vf_gradfun.o
libavfilter/x86/vf_gradfun.asm:52: warning: improperly calling multi-line macro `SETUP_STACK_POINTER' with 0 parameters [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:555: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:52: warning: improperly calling multi-line macro `ALLOC_STACK' with 0 parameters [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:558: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:52: warning: dropping trailing empty parameter in call to multi-line macro `DEFINE_ARGS_INTERNAL' [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:560: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:70: warning: improperly calling multi-line macro `SETUP_STACK_POINTER' with 0 parameters [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:555: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:70: warning: improperly calling multi-line macro `ALLOC_STACK' with 0 parameters [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:558: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:70: warning: dropping trailing empty parameter in call to multi-line macro `DEFINE_ARGS_INTERNAL' [-w+pp-macro-params-legacy]
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:560: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:109: warning: improperly calling multi-line macro `SETUP_STACK_POINTER' with 0 parameters [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:555: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:109: warning: improperly calling multi-line macro `ALLOC_STACK' with 0 parameters [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:558: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:109: warning: dropping trailing empty parameter in call to multi-line macro `DEFINE_ARGS_INTERNAL' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:560: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:110: warning: improperly calling multi-line macro `SETUP_STACK_POINTER' with 0 parameters [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:555: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:110: warning: improperly calling multi-line macro `ALLOC_STACK' with 0 parameters [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:558: ... from macro `PROLOGUE' defined here
libavfilter/x86/vf_gradfun.asm:110: warning: dropping trailing empty parameter in call to multi-line macro `DEFINE_ARGS_INTERNAL' [-w+pp-macro-params-legacy]
libavfilter/x86/vf_gradfun.asm:84: ... from macro `BLUR_LINE' defined here
libavutil/x86/x86inc.asm:707: ... from macro `cglobal' defined here
libavutil/x86/x86inc.asm:742: ... from macro `cglobal_internal' defined here
libavutil/x86/x86inc.asm:560: ... from macro `PROLOGUE' defined here
STRIP	libavfilter/x86/vf_gradfun.o
CC	libavfilter/x86/vf_gradfun_init.o
X86ASM	libavfilter/x86/vf_hflip.o
STRIP	libavfilter/x86/vf_hflip.o
CC	libavfilter/x86/vf_hflip_init.o
X86ASM	libavfilter/x86/vf_hqdn3d.o
STRIP	libavfilter/x86/vf_hqdn3d.o
CC	libavfilter/x86/vf_hqdn3d_init.o
X86ASM	libavfilter/x86/vf_idet.o
STRIP	libavfilter/x86/vf_idet.o
CC	libavfilter/x86/vf_idet_init.o
X86ASM	libavfilter/x86/vf_interlace.o
STRIP	libavfilter/x86/vf_interlace.o
X86ASM	libavfilter/x86/vf_limiter.o
STRIP	libavfilter/x86/vf_limiter.o
CC	libavfilter/x86/vf_limiter_init.o
X86ASM	libavfilter/x86/vf_maskedclamp.o
STRIP	libavfilter/x86/vf_maskedclamp.o
CC	libavfilter/x86/vf_maskedclamp_init.o
X86ASM	libavfilter/x86/vf_maskedmerge.o
STRIP	libavfilter/x86/vf_maskedmerge.o
CC	libavfilter/x86/vf_maskedmerge_init.o
CC	libavfilter/x86/vf_noise.o
X86ASM	libavfilter/x86/vf_overlay.o
STRIP	libavfilter/x86/vf_overlay.o
CC	libavfilter/x86/vf_overlay_init.o
X86ASM	libavfilter/x86/vf_pp7.o
STRIP	libavfilter/x86/vf_pp7.o
CC	libavfilter/x86/vf_pp7_init.o
X86ASM	libavfilter/x86/vf_psnr.o
STRIP	libavfilter/x86/vf_psnr.o
CC	libavfilter/x86/vf_psnr_init.o
X86ASM	libavfilter/x86/vf_pullup.o
STRIP	libavfilter/x86/vf_pullup.o
CC	libavfilter/x86/vf_pullup_init.o
X86ASM	libavfilter/x86/vf_removegrain.o
STRIP	libavfilter/x86/vf_removegrain.o
CC	libavfilter/x86/vf_removegrain_init.o
CC	libavfilter/x86/vf_spp.o
X86ASM	libavfilter/x86/vf_ssim.o
STRIP	libavfilter/x86/vf_ssim.o
CC	libavfilter/x86/vf_ssim_init.o
X86ASM	libavfilter/x86/vf_stereo3d.o
STRIP	libavfilter/x86/vf_stereo3d.o
CC	libavfilter/x86/vf_stereo3d_init.o
X86ASM	libavfilter/x86/vf_threshold.o
STRIP	libavfilter/x86/vf_threshold.o
CC	libavfilter/x86/vf_threshold_init.o
CC	libavfilter/x86/vf_tinterlace_init.o
X86ASM	libavfilter/x86/vf_transpose.o
STRIP	libavfilter/x86/vf_transpose.o
CC	libavfilter/x86/vf_transpose_init.o
X86ASM	libavfilter/x86/vf_v360.o
STRIP	libavfilter/x86/vf_v360.o
CC	libavfilter/x86/vf_v360_init.o
X86ASM	libavfilter/x86/vf_w3fdif.o
STRIP	libavfilter/x86/vf_w3fdif.o
CC	libavfilter/x86/vf_w3fdif_init.o
X86ASM	libavfilter/x86/vf_yadif.o
STRIP	libavfilter/x86/vf_yadif.o
CC	libavfilter/x86/vf_yadif_init.o
X86ASM	libavfilter/x86/yadif-10.o
STRIP	libavfilter/x86/yadif-10.o
X86ASM	libavfilter/x86/yadif-16.o
STRIP	libavfilter/x86/yadif-16.o
CC	libavfilter/yadif_common.o
AR	libavfilter/libavfilter.a
CC	libavformat/3dostr.o
CC	libavformat/4xm.o
CC	libavformat/a64.o
CC	libavformat/aacdec.o
CC	libavformat/aadec.o
libavformat/aadec.c: In function ‘aa_read_header’:
libavformat/aadec.c:116:13: warning: ‘__builtin_strncpy’ output may be truncated copying 63 bytes from a string of length 127 [-Wstringop-truncation]
  116 |             strncpy(codec_name, val, sizeof(codec_name) - 1);
      |             ^
CC	libavformat/ac3dec.o
CC	libavformat/acm.o
CC	libavformat/act.o
CC	libavformat/adp.o
CC	libavformat/ads.o
CC	libavformat/adtsenc.o
./libavcodec/x86/mathops.h: Assembler messages:
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
./libavcodec/x86/mathops.h:125: Error: operand type mismatch for `shr'
make: *** [ffbuild/common.mak:59: libavformat/adtsenc.o] Error 1
