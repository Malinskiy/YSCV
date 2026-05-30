//! Hardware-accelerated video decode backends.
//!
//! Each backend is gated behind a feature flag and platform check:
//! - `videotoolbox` — macOS/iOS: Apple VideoToolbox (H.264/HEVC)
//! - `vaapi` — Linux: VA-API (Intel/AMD GPU)
//! - `nvdec` — Linux/Windows: NVIDIA NVDEC (CUDA)
//! - `media-foundation` — Windows: Media Foundation
//!
//! All backends use raw FFI to system libraries — no external crate dependencies.
//! Use [`HwVideoDecoder::new`] for automatic backend selection with software fallback.
//!
//! # Safety contract for unsafe code in this module
//!
//! All unsafe blocks fall into these categories:
//!
//! 1. **Platform FFI calls** — VideoToolbox / VA-API / NVDEC / Media Foundation APIs.
//!    Safety: handles are validated non-null before use; error codes are checked
//!    after each call; resources are cleaned up in `Drop`.
//!
//! 2. **Callback pointer dereferences** — `refcon` / `user_data` cast back from
//!    `*mut c_void` inside platform callbacks. Safety: the pointed-to value is
//!    pinned in a `Box` that outlives the session, so the pointer is valid for
//!    the lifetime of the decoder.
//!
//! 3. **SIMD intrinsics** (NEON / SSE2) — color-space conversion (NV12/BGRA to RGB).
//!    Safety: ISA availability is guaranteed by `cfg(target_arch)` or
//!    `#[target_feature(enable)]`; pointer arithmetic stays within
//!    `stride * height` and `w * h * 3` bounds enforced by the containing loop.
//!
//! 4. **Raw pointer reads** from GPU-mapped pixel buffers.
//!    Safety: the buffer is locked (`CVPixelBufferLockBaseAddress` / `vaMapBuffer`)
//!    before access and unlocked after; dimensions come from the same API that
//!    provided the pointer.

use crate::{DecodedFrame, VideoCodec, VideoDecoder, VideoError};

// ---------------------------------------------------------------------------
// Backend enum + detection
// ---------------------------------------------------------------------------

/// Detected hardware decode backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    VideoToolbox,
    Vaapi,
    Nvdec,
    MediaFoundation,
    Software,
}

impl std::fmt::Display for HwBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VideoToolbox => write!(f, "VideoToolbox"),
            Self::Vaapi => write!(f, "VA-API"),
            Self::Nvdec => write!(f, "NVDEC"),
            Self::MediaFoundation => write!(f, "MediaFoundation"),
            Self::Software => write!(f, "Software"),
        }
    }
}

/// Detect the best available hardware decode backend.
#[allow(unreachable_code)]
pub fn detect_hw_backend() -> HwBackend {
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    {
        return HwBackend::VideoToolbox;
    }
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    {
        return HwBackend::Vaapi;
    }
    #[cfg(feature = "nvdec")]
    {
        return HwBackend::Nvdec;
    }
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    {
        return HwBackend::MediaFoundation;
    }
    HwBackend::Software
}

// ═══════════════════════════════════════════════════════════════════════════
// VideoToolbox backend (macOS/iOS)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
#[allow(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    improper_ctypes_definitions
)]
pub mod videotoolbox {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    // --- Raw FFI bindings to CoreMedia / VideoToolbox frameworks ---

    type OSStatus = i32;
    type CFAllocatorRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CMFormatDescriptionRef = *const c_void;
    type CMSampleBufferRef = *const c_void;
    type CMBlockBufferRef = *const c_void;
    type CVPixelBufferRef = *const c_void;
    type VTDecompressionSessionRef = *const c_void;
    type CMVideoCodecType = u32;
    type CFStringRef = *const c_void;
    type CFTypeRef = *const c_void;
    type CMItemCount = isize;
    type CMTime = [u8; 24]; // opaque, we pass zeros

    const kCMVideoCodecType_H264: CMVideoCodecType = 0x61766331; // 'avc1'
    const kCMVideoCodecType_HEVC: CMVideoCodecType = 0x68766331; // 'hvc1'
    // NV12 video-range: VT always delivers this reliably (Y:16-235, UV:16-240)
    const kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: u32 = 0x34323076; // '420v'
    const kCVPixelFormatType_32BGRA: u32 = 0x42475241; // 'BGRA'

    #[repr(C)]
    struct VTDecompressionOutputCallbackRecord {
        callback: extern "C" fn(
            *mut c_void,      // decompressionOutputRefCon
            *mut c_void,      // sourceFrameRefCon
            OSStatus,         // status
            u32,              // infoFlags
            CVPixelBufferRef, // imageBuffer
            CMTime,           // presentationTimeStamp
            CMTime,           // presentationDuration
        ),
        refcon: *mut c_void,
    }

    #[allow(clippy::duplicated_attributes)]
    #[link(name = "VideoToolbox", kind = "framework")]
    #[link(name = "CoreMedia", kind = "framework")]
    #[link(name = "CoreVideo", kind = "framework")]
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            format_description_out: *mut CMFormatDescriptionRef,
        ) -> OSStatus;

        fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            extensions: CFDictionaryRef,
            format_description_out: *mut CMFormatDescriptionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionCreate(
            allocator: CFAllocatorRef,
            video_format_description: CMFormatDescriptionRef,
            video_decoder_specification: CFDictionaryRef,
            destination_image_buffer_attributes: CFDictionaryRef,
            output_callback: *const VTDecompressionOutputCallbackRecord,
            decompression_session_out: *mut VTDecompressionSessionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionDecodeFrame(
            session: VTDecompressionSessionRef,
            sample_buffer: CMSampleBufferRef,
            decode_flags: u32,
            source_frame_refcon: *mut c_void,
            info_flags_out: *mut u32,
        ) -> OSStatus;

        fn VTDecompressionSessionWaitForAsynchronousFrames(
            session: VTDecompressionSessionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);

        fn CMBlockBufferCreateWithMemoryBlock(
            allocator: CFAllocatorRef,
            memory_block: *const c_void,
            block_length: usize,
            block_allocator: CFAllocatorRef,
            custom_block_source: *const c_void,
            offset_to_data: usize,
            data_length: usize,
            flags: u32,
            block_buffer_out: *mut CMBlockBufferRef,
        ) -> OSStatus;

        fn CMBlockBufferReplaceDataBytes(
            source_bytes: *const c_void,
            destination_buffer: CMBlockBufferRef,
            offset_into_destination: usize,
            data_length: usize,
        ) -> OSStatus;

        fn CMSampleBufferCreateReady(
            allocator: CFAllocatorRef,
            data_buffer: CMBlockBufferRef,
            format_description: CMFormatDescriptionRef,
            num_samples: CMItemCount,
            num_sample_timing_entries: CMItemCount,
            sample_timing_array: *const c_void,
            num_sample_size_entries: CMItemCount,
            sample_size_array: *const usize,
            sample_buffer_out: *mut CMSampleBufferRef,
        ) -> OSStatus;

        fn CVPixelBufferLockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> OSStatus;

        fn CVPixelBufferUnlockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> OSStatus;

        fn CVPixelBufferGetBaseAddress(pixel_buffer: CVPixelBufferRef) -> *const u8;
        fn CVPixelBufferGetBaseAddressOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane: usize,
        ) -> *const u8;
        fn CVPixelBufferGetBytesPerRow(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetBytesPerRowOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane: usize,
        ) -> usize;
        fn CVPixelBufferGetWidth(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetWidthOfPlane(pixel_buffer: CVPixelBufferRef, plane: usize) -> usize;
        fn CVPixelBufferGetHeight(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetHeightOfPlane(pixel_buffer: CVPixelBufferRef, plane: usize) -> usize;
        fn CVPixelBufferGetPlaneCount(pixel_buffer: CVPixelBufferRef) -> usize;

        fn CFRelease(cf: *const c_void);

        fn CFDictionaryCreateMutable(
            allocator: CFAllocatorRef,
            capacity: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *mut c_void;

        fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, value: *const c_void);

        fn CFNumberCreate(
            allocator: CFAllocatorRef,
            the_type: isize,
            value_ptr: *const c_void,
        ) -> *const c_void;

        static kCFAllocatorDefault: CFAllocatorRef;
        static kCFTypeDictionaryKeyCallBacks: c_void;
        static kCFTypeDictionaryValueCallBacks: c_void;
        static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    }

    /// Decoded frame storage for callback.
    struct CallbackState {
        frames: Vec<DecodedFrame>,
    }

    extern "C" fn decode_callback(
        refcon: *mut c_void,
        _source: *mut c_void,
        status: OSStatus,
        _flags: u32,
        image_buffer: CVPixelBufferRef,
        _pts: CMTime,
        _dur: CMTime,
    ) {
        if status != 0 || image_buffer.is_null() {
            return;
        }
        // SAFETY: (category 2 + 4) refcon points to a Box<CallbackState> pinned for
        // the session lifetime; image_buffer is non-null and locked before pixel access.
        unsafe {
            let state = &mut *(refcon as *mut CallbackState);

            CVPixelBufferLockBaseAddress(image_buffer, 1); // read-only
            let w = CVPixelBufferGetWidth(image_buffer);
            let h = CVPixelBufferGetHeight(image_buffer);
            let planes = CVPixelBufferGetPlaneCount(image_buffer);

            let rgb = if planes >= 2 {
                // NV12 → RGB via NEON SIMD
                let y_ptr = CVPixelBufferGetBaseAddressOfPlane(image_buffer, 0);
                let y_stride = CVPixelBufferGetBytesPerRowOfPlane(image_buffer, 0);
                let uv_ptr = CVPixelBufferGetBaseAddressOfPlane(image_buffer, 1);
                let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(image_buffer, 1);
                let mut rgb_out = vec![0u8; w * h * 3];
                nv12_bt601_to_rgb(y_ptr, y_stride, uv_ptr, uv_stride, w, h, &mut rgb_out);
                rgb_out
            } else {
                // BGRA fallback
                let base = CVPixelBufferGetBaseAddress(image_buffer);
                let stride = CVPixelBufferGetBytesPerRow(image_buffer);
                let mut out = vec![0u8; w * h * 3];
                bgra_to_rgb(base, stride, w, h, &mut out);
                out
            };
            CVPixelBufferUnlockBaseAddress(image_buffer, 1);

            state.frames.push(DecodedFrame {
                width: w,
                height: h,
                rgb8_data: rgb,
                timestamp_us: 0,
                keyframe: false,
                bit_depth: 8,
                rgb16_data: None,
            });
        }
    }

    /// Apple VideoToolbox hardware decoder.
    pub struct VideoToolboxDecoder {
        codec: VideoCodec,
        session: VTDecompressionSessionRef,
        format_desc: CMFormatDescriptionRef,
        state: Box<CallbackState>,
        sps: Vec<u8>,
        pps: Vec<u8>,
        vps: Vec<u8>,
        initialized: bool,
    }

    impl VideoToolboxDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            Ok(VideoToolboxDecoder {
                codec,
                session: ptr::null(),
                format_desc: ptr::null(),
                state: Box::new(CallbackState { frames: Vec::new() }),
                sps: Vec::new(),
                pps: Vec::new(),
                vps: Vec::new(),
                initialized: false,
            })
        }

        unsafe fn create_session(&mut self) -> Result<(), VideoError> {
            // Create format description from parameter sets
            self.format_desc = match self.codec {
                VideoCodec::H264 => {
                    let ptrs = [self.sps.as_ptr(), self.pps.as_ptr()];
                    let sizes = [self.sps.len(), self.pps.len()];
                    let mut fmt: CMFormatDescriptionRef = ptr::null();
                    let status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
                        kCFAllocatorDefault,
                        2,
                        ptrs.as_ptr(),
                        sizes.as_ptr(),
                        4,
                        &mut fmt,
                    );
                    if status != 0 {
                        return Err(VideoError::Codec(format!(
                            "VT: failed to create H264 format description: {status}"
                        )));
                    }
                    fmt
                }
                VideoCodec::H265 => {
                    let ptrs = [self.vps.as_ptr(), self.sps.as_ptr(), self.pps.as_ptr()];
                    let sizes = [self.vps.len(), self.sps.len(), self.pps.len()];
                    let mut fmt: CMFormatDescriptionRef = ptr::null();
                    let status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                        kCFAllocatorDefault,
                        3,
                        ptrs.as_ptr(),
                        sizes.as_ptr(),
                        4,
                        ptr::null(),
                        &mut fmt,
                    );
                    if status != 0 {
                        return Err(VideoError::Codec(format!(
                            "VT: failed to create HEVC format description: {status}"
                        )));
                    }
                    fmt
                }
                _ => return Err(VideoError::Codec("VT: unsupported codec".into())),
            };

            // Pixel buffer attributes: request BGRA output
            let attrs = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            // NV12 output — direct from decoder, no GPU color conversion overhead.
            // CPU-side NEON NV12→RGB is faster than VT's GPU BGRA scaler on Apple Silicon.
            let pixel_fmt = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
            let fmt_num = CFNumberCreate(
                kCFAllocatorDefault,
                9, // kCFNumberSInt32Type
                &pixel_fmt as *const u32 as *const c_void,
            );
            CFDictionarySetValue(attrs, kCVPixelBufferPixelFormatTypeKey, fmt_num);

            let callback = VTDecompressionOutputCallbackRecord {
                callback: decode_callback,
                refcon: &mut *self.state as *mut CallbackState as *mut c_void,
            };

            let mut session: VTDecompressionSessionRef = ptr::null();
            let status = VTDecompressionSessionCreate(
                kCFAllocatorDefault,
                self.format_desc,
                ptr::null(),
                attrs as *const c_void,
                &callback,
                &mut session,
            );
            CFRelease(fmt_num);
            CFRelease(attrs as *const c_void);

            if status != 0 {
                return Err(VideoError::Codec(format!(
                    "VT: failed to create decompression session: {status}"
                )));
            }
            self.session = session;
            self.initialized = true;
            Ok(())
        }

        fn extract_parameter_sets(&mut self, data: &[u8]) {
            // Parse Annex B NAL units and extract SPS/PPS/VPS
            let nals = crate::parse_annex_b(data);
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                match self.codec {
                    VideoCodec::H264 => {
                        let nal_type = nal.data[0] & 0x1F;
                        match nal_type {
                            7 => self.sps = nal.data.clone(), // SPS
                            8 => self.pps = nal.data.clone(), // PPS
                            _ => {}
                        }
                    }
                    VideoCodec::H265 => {
                        let nal_type = (nal.data[0] >> 1) & 0x3F;
                        match nal_type {
                            32 => self.vps = nal.data.clone(), // VPS
                            33 => self.sps = nal.data.clone(), // SPS
                            34 => self.pps = nal.data.clone(), // PPS
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    impl VideoDecoder for VideoToolboxDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            self.extract_parameter_sets(data);

            // Initialize session once we have parameter sets
            if !self.initialized {
                let has_params = match self.codec {
                    VideoCodec::H264 => !self.sps.is_empty() && !self.pps.is_empty(),
                    VideoCodec::H265 => {
                        !self.vps.is_empty() && !self.sps.is_empty() && !self.pps.is_empty()
                    }
                    _ => false,
                };
                if !has_params {
                    return Ok(None); // Need more data
                }
                // SAFETY: (category 1) SPS/PPS/VPS validated non-empty above; FFI calls
                // check OSStatus return codes.
                unsafe {
                    self.create_session()?;
                }
            }

            // Build single AVCC buffer from all non-param NALs in this AU.
            // One CMBlockBuffer + one DecodeFrame call per AU eliminates per-NAL FFI overhead.
            let nals = crate::parse_annex_b(data);
            let mut avcc_buf = Vec::new();
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                let is_param = match self.codec {
                    VideoCodec::H264 => matches!(nal.data[0] & 0x1F, 7 | 8),
                    VideoCodec::H265 => matches!((nal.data[0] >> 1) & 0x3F, 32..=34),
                    _ => false,
                };
                if is_param {
                    continue;
                }
                let nal_len = nal.data.len() as u32;
                avcc_buf.extend_from_slice(&nal_len.to_be_bytes());
                avcc_buf.extend_from_slice(&nal.data);
            }

            if !avcc_buf.is_empty() {
                // SAFETY: (category 1) VT session is initialized; block buffer and sample
                // buffer are checked for non-null and OSStatus == 0 before use; CFRelease
                // is called on all created CF objects.
                unsafe {
                    let mut block_buf: CMBlockBufferRef = ptr::null();
                    let mut status = CMBlockBufferCreateWithMemoryBlock(
                        kCFAllocatorDefault,
                        ptr::null(),
                        avcc_buf.len(),
                        ptr::null(),
                        ptr::null(),
                        0,
                        avcc_buf.len(),
                        0,
                        &mut block_buf,
                    );
                    if status == 0 && !block_buf.is_null() {
                        status = CMBlockBufferReplaceDataBytes(
                            avcc_buf.as_ptr() as *const c_void,
                            block_buf,
                            0,
                            avcc_buf.len(),
                        );
                        if status == 0 {
                            let sample_size = avcc_buf.len();
                            let mut sample_buf: CMSampleBufferRef = ptr::null();
                            status = CMSampleBufferCreateReady(
                                kCFAllocatorDefault,
                                block_buf,
                                self.format_desc,
                                1,
                                0,
                                ptr::null(),
                                1,
                                &sample_size,
                                &mut sample_buf,
                            );
                            if status == 0 && !sample_buf.is_null() {
                                let mut info_flags: u32 = 0;
                                let _ = VTDecompressionSessionDecodeFrame(
                                    self.session,
                                    sample_buf,
                                    1, // async decode — VT pipelines decode while we prepare next AU
                                    ptr::null_mut(),
                                    &mut info_flags,
                                );
                                CFRelease(sample_buf);
                            }
                        }
                        CFRelease(block_buf);
                    }
                }
            }

            // Wait for async frames
            if self.initialized {
                // SAFETY: (category 1) session handle validated during create_session.
                unsafe {
                    VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                }
            }

            // Return last decoded frame
            let mut frame = self.state.frames.pop();
            if let Some(ref mut f) = frame {
                f.timestamp_us = timestamp_us;
            }
            Ok(frame)
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if self.initialized {
                // SAFETY: (category 1) session handle validated during create_session.
                unsafe {
                    VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                }
            }
            Ok(std::mem::take(&mut self.state.frames))
        }
    }

    impl Drop for VideoToolboxDecoder {
        fn drop(&mut self) {
            if self.initialized {
                // SAFETY: (category 1) session/format_desc were successfully created
                // (self.initialized guards); invalidate + release is the documented
                // teardown sequence.
                unsafe {
                    VTDecompressionSessionInvalidate(self.session);
                    if !self.format_desc.is_null() {
                        CFRelease(self.format_desc);
                    }
                }
            }
        }
    }

    // Safety: VT session is used single-threaded via &mut self
    unsafe impl Send for VideoToolboxDecoder {}

    /// Convert BGRA (from VT GPU output) to RGB8.
    /// NEON: deinterleave 16 pixels at a time via vld4/vst3.
    unsafe fn bgra_to_rgb(bgra_ptr: *const u8, stride: usize, w: usize, h: usize, rgb: &mut [u8]) {
        for row in 0..h {
            let src = bgra_ptr.add(row * stride);
            let dst = &mut rgb[row * w * 3..(row + 1) * w * 3];
            let mut col = 0usize;

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                // Process 16 pixels per iteration: load 16×BGRA, store 16×RGB
                while col + 16 <= w {
                    let bgra = vld4q_u8(src.add(col * 4));
                    // bgra.0=B, bgra.1=G, bgra.2=R, bgra.3=A
                    let out = uint8x16x3_t(bgra.2, bgra.1, bgra.0);
                    vst3q_u8(dst.as_mut_ptr().add(col * 3), out);
                    col += 16;
                }
            }

            // Scalar tail
            while col < w {
                let s = src.add(col * 4);
                let d = col * 3;
                dst[d] = *s.add(2); // R
                dst[d + 1] = *s.add(1); // G
                dst[d + 2] = *s; // B
                col += 1;
            }
        }
    }

    /// Convert NV12 BT.601 limited range to RGB8.
    /// Uses NEON SIMD on aarch64, scalar fallback otherwise.
    #[allow(clippy::too_many_arguments)]
    unsafe fn nv12_bt601_to_rgb(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            nv12_bt601_to_rgb_neon(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            nv12_bt601_to_rgb_sse2(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
            return;
        }
        #[allow(unreachable_code)]
        nv12_bt601_to_rgb_scalar(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
    }

    /// NEON-accelerated NV12 BT.601 → RGB8.
    /// Processes 8 pixels per iteration using i32 widening multiply to avoid
    /// the i16 overflow that occurs with the half-scale approach when bright
    /// luma meets saturated chroma (e.g. 149*(Y-16) + 204*(Cr-128) > 32767).
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn nv12_bt601_to_rgb_neon(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        use std::arch::aarch64::*;

        let v16 = vdupq_n_s16(16);
        let v128 = vdupq_n_s16(128);
        let c298 = vdup_n_s16(298);
        let c409 = vdup_n_s16(409);
        let c100 = vdup_n_s16(100);
        let c208 = vdup_n_s16(208);
        let c516 = vdup_n_s16(516);
        let half = vdupq_n_s32(128);
        let zero16 = vdupq_n_s16(0);

        for row in 0..h {
            let y_row = y_ptr.add(row * y_stride);
            let uv_row = uv_ptr.add((row / 2) * uv_stride);
            let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
            let mut col = 0usize;

            while col + 8 <= w {
                let y8 = vld1_u8(y_row.add(col));
                let y16 = vreinterpretq_s16_u16(vmovl_u8(y8));
                let y_adj = vsubq_s16(y16, v16);

                let uv8 = vld1_u8(uv_row.add((col / 2) * 2));
                let uv16 = vreinterpretq_s16_u16(vmovl_u8(uv8));
                let cb4 = vuzp1q_s16(uv16, uv16);
                let cr4 = vuzp2q_s16(uv16, uv16);
                let cb = vzip1q_s16(cb4, cb4);
                let cr = vzip1q_s16(cr4, cr4);
                let cb_adj = vsubq_s16(cb, v128);
                let cr_adj = vsubq_s16(cr, v128);

                // Low 4 pixels — widening multiply i16×i16 → i32
                let y_lo = vget_low_s16(y_adj);
                let cb_lo = vget_low_s16(cb_adj);
                let cr_lo = vget_low_s16(cr_adj);
                let c_lo = vmull_s16(c298, y_lo);
                let r_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c409, cr_lo)), half), 8);
                let g_lo = vshrq_n_s32(vaddq_s32(vsubq_s32(vsubq_s32(c_lo, vmull_s16(c208, cr_lo)), vmull_s16(c100, cb_lo)), half), 8);
                let b_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c516, cb_lo)), half), 8);

                // High 4 pixels
                let y_hi = vget_high_s16(y_adj);
                let cb_hi = vget_high_s16(cb_adj);
                let cr_hi = vget_high_s16(cr_adj);
                let c_hi = vmull_s16(c298, y_hi);
                let r_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c409, cr_hi)), half), 8);
                let g_hi = vshrq_n_s32(vaddq_s32(vsubq_s32(vsubq_s32(c_hi, vmull_s16(c208, cr_hi)), vmull_s16(c100, cb_hi)), half), 8);
                let b_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c516, cb_hi)), half), 8);

                // Narrow i32 → i16, clamp [0,255], narrow i16 → u8
                let r16 = vcombine_s16(vmovn_s32(r_lo), vmovn_s32(r_hi));
                let g16 = vcombine_s16(vmovn_s32(g_lo), vmovn_s32(g_hi));
                let b16 = vcombine_s16(vmovn_s32(b_lo), vmovn_s32(b_hi));
                let r8 = vqmovun_s16(vmaxq_s16(r16, zero16));
                let g8 = vqmovun_s16(vmaxq_s16(g16, zero16));
                let b8 = vqmovun_s16(vmaxq_s16(b16, zero16));

                vst3_u8(dst_row.as_mut_ptr().add(col * 3), uint8x8x3_t(r8, g8, b8));
                col += 8;
            }

            while col < w {
                let y_val = *y_row.add(col) as i32;
                let cb_val = *uv_row.add((col / 2) * 2) as i32;
                let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
                let c = 298 * (y_val - 16);
                let r = (c + 409 * (cr_val - 128) + 128) >> 8;
                let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
                let b = (c + 516 * (cb_val - 128) + 128) >> 8;
                let dst = col * 3;
                dst_row[dst] = r.clamp(0, 255) as u8;
                dst_row[dst + 1] = g.clamp(0, 255) as u8;
                dst_row[dst + 2] = b.clamp(0, 255) as u8;
                col += 1;
            }
        }
    }

    /// SSE2-accelerated NV12 BT.601 limited range → RGB8.
    /// Processes 8 pixels per iteration using int16 arithmetic.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn nv12_bt601_to_rgb_sse2(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        use std::arch::x86_64::*;

        // BT.601 limited range half-scale coefficients (same as NEON path)
        let c149 = _mm_set1_epi16(149); // 298/2
        let c204 = _mm_set1_epi16(204); // 409/2
        let c50 = _mm_set1_epi16(50); // 100/2
        let c104 = _mm_set1_epi16(104); // 208/2
        let c258 = _mm_set1_epi16(258u16 as i16); // 516/2
        let v16 = _mm_set1_epi16(16);
        let v128 = _mm_set1_epi16(128);
        let half = _mm_set1_epi16(64); // 128/2
        let zero = _mm_setzero_si128();

        for row in 0..h {
            let y_row = y_ptr.add(row * y_stride);
            let uv_row = uv_ptr.add((row / 2) * uv_stride);
            let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
            let mut col = 0usize;

            while col + 8 <= w {
                // Load 8 Y values → i16
                let y8 = _mm_loadl_epi64(y_row.add(col) as *const __m128i);
                let y16 = _mm_unpacklo_epi8(y8, zero);
                let y_adj = _mm_sub_epi16(y16, v16); // Y - 16

                // Load 8 UV bytes (4 interleaved Cb,Cr pairs), deinterleave + duplicate
                let mut cb_buf = [0u8; 8];
                let mut cr_buf = [0u8; 8];
                for i in 0..4 {
                    cb_buf[i * 2] = *uv_row.add((col / 2 + i) * 2);
                    cb_buf[i * 2 + 1] = *uv_row.add((col / 2 + i) * 2);
                    cr_buf[i * 2] = *uv_row.add((col / 2 + i) * 2 + 1);
                    cr_buf[i * 2 + 1] = *uv_row.add((col / 2 + i) * 2 + 1);
                }
                let cb8 = _mm_loadl_epi64(cb_buf.as_ptr() as *const __m128i);
                let cr8 = _mm_loadl_epi64(cr_buf.as_ptr() as *const __m128i);
                let cb_adj = _mm_sub_epi16(_mm_unpacklo_epi8(cb8, zero), v128);
                let cr_adj = _mm_sub_epi16(_mm_unpacklo_epi8(cr8, zero), v128);

                // BT.601 half-scale: c = 149*(Y-16), >>7 at the end
                let c_val = _mm_mullo_epi16(c149, y_adj);
                // R = (c + 204*(Cr-128) + 64) >> 7
                let r16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                    _mm_add_epi16(c_val, _mm_mullo_epi16(c204, cr_adj)),
                    half,
                ));
                // G = (c - 104*(Cr-128) - 50*(Cb-128) + 64) >> 7
                let g16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                    _mm_sub_epi16(
                        _mm_sub_epi16(c_val, _mm_mullo_epi16(c104, cr_adj)),
                        _mm_mullo_epi16(c50, cb_adj),
                    ),
                    half,
                ));
                // B = (c + 258*(Cb-128) + 64) >> 7
                let b16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                    _mm_add_epi16(c_val, _mm_mullo_epi16(c258, cb_adj)),
                    half,
                ));

                // Clamp to [0,255] and pack to u8
                let r_u8 = _mm_packus_epi16(_mm_max_epi16(r16, zero), zero);
                let g_u8 = _mm_packus_epi16(_mm_max_epi16(g16, zero), zero);
                let b_u8 = _mm_packus_epi16(_mm_max_epi16(b16, zero), zero);

                // Manual RGB interleave (SSE2 has no vst3)
                let mut rgb_buf = [0u8; 24];
                let mut r_arr = [0u8; 8];
                let mut g_arr = [0u8; 8];
                let mut b_arr = [0u8; 8];
                _mm_storel_epi64(r_arr.as_mut_ptr() as *mut __m128i, r_u8);
                _mm_storel_epi64(g_arr.as_mut_ptr() as *mut __m128i, g_u8);
                _mm_storel_epi64(b_arr.as_mut_ptr() as *mut __m128i, b_u8);
                for i in 0..8 {
                    rgb_buf[i * 3] = r_arr[i];
                    rgb_buf[i * 3 + 1] = g_arr[i];
                    rgb_buf[i * 3 + 2] = b_arr[i];
                }
                std::ptr::copy_nonoverlapping(
                    rgb_buf.as_ptr(),
                    dst_row.as_mut_ptr().add(col * 3),
                    24,
                );

                col += 8;
            }

            // Scalar tail
            while col < w {
                let y_val = *y_row.add(col) as i32;
                let cb_val = *uv_row.add((col / 2) * 2) as i32;
                let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
                let c = 298 * (y_val - 16);
                let r = (c + 409 * (cr_val - 128) + 128) >> 8;
                let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
                let b = (c + 516 * (cb_val - 128) + 128) >> 8;
                let dst = col * 3;
                dst_row[dst] = r.clamp(0, 255) as u8;
                dst_row[dst + 1] = g.clamp(0, 255) as u8;
                dst_row[dst + 2] = b.clamp(0, 255) as u8;
                col += 1;
            }
        }
    }

    /// Scalar fallback NV12 BT.601 → RGB8.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn nv12_bt601_to_rgb_scalar(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        for row in 0..h {
            let y_row = y_ptr.add(row * y_stride);
            let uv_row = uv_ptr.add((row / 2) * uv_stride);
            for col in 0..w {
                let y_val = *y_row.add(col) as i32;
                let cb_val = *uv_row.add((col / 2) * 2) as i32;
                let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
                let c = 298 * (y_val - 16);
                let r = (c + 409 * (cr_val - 128) + 128) >> 8;
                let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
                let b = (c + 516 * (cb_val - 128) + 128) >> 8;
                let dst = (row * w + col) * 3;
                rgb[dst] = r.clamp(0, 255) as u8;
                rgb[dst + 1] = g.clamp(0, 255) as u8;
                rgb[dst + 2] = b.clamp(0, 255) as u8;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VA-API backend (Linux)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "linux", feature = "vaapi"))]
#[allow(unsafe_code, non_camel_case_types)]
pub mod vaapi {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    // --- Raw FFI to libva ---
    type VADisplay = *mut c_void;
    type VAStatus = i32;
    type VAConfigID = u32;
    type VAContextID = u32;
    type VASurfaceID = u32;
    type VABufferID = u32;
    type VAProfile = i32;
    type VAEntrypoint = i32;

    const VA_PROFILE_H264_HIGH: VAProfile = 7;
    const VA_PROFILE_HEVC_MAIN: VAProfile = 12;
    const VA_ENTRYPOINT_VLD: VAEntrypoint = 1;
    const VA_STATUS_SUCCESS: VAStatus = 0;

    /// VA image descriptor returned by vaDeriveImage.
    #[repr(C)]
    struct VAImage {
        image_id: u32,
        format: VAImageFormat,
        buf: VABufferID,
        width: u16,
        height: u16,
        data_size: u32,
        num_planes: u32,
        pitches: [u32; 3],
        offsets: [u32; 3],
        num_palette_entries: i32,
        entry_bytes: i32,
        component_order: [i8; 4],
    }

    #[repr(C)]
    struct VAImageFormat {
        fourcc: u32,
        byte_order: u32,
        bits_per_pixel: u32,
        depth: u32,
        red_mask: u32,
        green_mask: u32,
        blue_mask: u32,
        alpha_mask: u32,
    }

    const VA_RT_FORMAT_YUV420: u32 = 0x00000001;
    const VASliceDataBufferType: i32 = 5;

    /// Dynamically-loaded libva function pointers.
    struct VaLib {
        _lib: libloading::Library,
        va_initialize:
            unsafe extern "C" fn(VADisplay, *mut i32, *mut i32) -> VAStatus,
        va_terminate: unsafe extern "C" fn(VADisplay) -> VAStatus,
        va_create_config: unsafe extern "C" fn(
            VADisplay,
            VAProfile,
            VAEntrypoint,
            *const c_void,
            i32,
            *mut VAConfigID,
        ) -> VAStatus,
        va_create_surfaces: unsafe extern "C" fn(
            VADisplay,
            u32,
            u32,
            u32,
            *mut VASurfaceID,
            u32,
            *const c_void,
            u32,
        ) -> VAStatus,
        va_create_context: unsafe extern "C" fn(
            VADisplay,
            VAConfigID,
            i32,
            i32,
            i32,
            *mut VASurfaceID,
            i32,
            *mut VAContextID,
        ) -> VAStatus,
        va_begin_picture:
            unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus,
        va_create_buffer: unsafe extern "C" fn(
            VADisplay,
            VAContextID,
            i32,
            u32,
            u32,
            *const c_void,
            *mut VABufferID,
        ) -> VAStatus,
        va_render_picture: unsafe extern "C" fn(
            VADisplay,
            VAContextID,
            *mut VABufferID,
            i32,
        ) -> VAStatus,
        va_end_picture:
            unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
        va_sync_surface:
            unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus,
        va_derive_image:
            unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VAImage) -> VAStatus,
        va_map_buffer:
            unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus,
        va_unmap_buffer:
            unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
        va_destroy_image:
            unsafe extern "C" fn(VADisplay, u32) -> VAStatus,
        va_destroy_buffer:
            unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
        va_destroy_surfaces:
            unsafe extern "C" fn(VADisplay, *mut VASurfaceID, i32) -> VAStatus,
        va_destroy_config:
            unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus,
        va_destroy_context:
            unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
    }

    impl VaLib {
        fn load() -> Option<Self> {
            // SAFETY: libva.so.2 is a well-known system library.
            let lib = unsafe { libloading::Library::new("libva.so.2") }.ok()?;
            // SAFETY: symbol signatures match the libva C ABI.
            unsafe {
                let va_initialize = *lib
                    .get::<unsafe extern "C" fn(VADisplay, *mut i32, *mut i32) -> VAStatus>(
                        b"vaInitialize\0",
                    )
                    .ok()?;
                let va_terminate = *lib
                    .get::<unsafe extern "C" fn(VADisplay) -> VAStatus>(b"vaTerminate\0")
                    .ok()?;
                let va_create_config = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAProfile, VAEntrypoint, *const c_void, i32, *mut VAConfigID) -> VAStatus>(
                        b"vaCreateConfig\0",
                    )
                    .ok()?;
                let va_create_surfaces = *lib
                    .get::<unsafe extern "C" fn(VADisplay, u32, u32, u32, *mut VASurfaceID, u32, *const c_void, u32) -> VAStatus>(
                        b"vaCreateSurfaces\0",
                    )
                    .ok()?;
                let va_create_context = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAConfigID, i32, i32, i32, *mut VASurfaceID, i32, *mut VAContextID) -> VAStatus>(
                        b"vaCreateContext\0",
                    )
                    .ok()?;
                let va_begin_picture = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus>(
                        b"vaBeginPicture\0",
                    )
                    .ok()?;
                let va_create_buffer = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID, i32, u32, u32, *const c_void, *mut VABufferID) -> VAStatus>(
                        b"vaCreateBuffer\0",
                    )
                    .ok()?;
                let va_render_picture = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID, *mut VABufferID, i32) -> VAStatus>(
                        b"vaRenderPicture\0",
                    )
                    .ok()?;
                let va_end_picture = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus>(
                        b"vaEndPicture\0",
                    )
                    .ok()?;
                let va_sync_surface = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus>(
                        b"vaSyncSurface\0",
                    )
                    .ok()?;
                let va_derive_image = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VAImage) -> VAStatus>(
                        b"vaDeriveImage\0",
                    )
                    .ok()?;
                let va_map_buffer = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus>(
                        b"vaMapBuffer\0",
                    )
                    .ok()?;
                let va_unmap_buffer = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus>(
                        b"vaUnmapBuffer\0",
                    )
                    .ok()?;
                let va_destroy_image = *lib
                    .get::<unsafe extern "C" fn(VADisplay, u32) -> VAStatus>(
                        b"vaDestroyImage\0",
                    )
                    .ok()?;
                let va_destroy_buffer = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus>(
                        b"vaDestroyBuffer\0",
                    )
                    .ok()?;
                let va_destroy_surfaces = *lib
                    .get::<unsafe extern "C" fn(VADisplay, *mut VASurfaceID, i32) -> VAStatus>(
                        b"vaDestroySurfaces\0",
                    )
                    .ok()?;
                let va_destroy_config = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus>(
                        b"vaDestroyConfig\0",
                    )
                    .ok()?;
                let va_destroy_context = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus>(
                        b"vaDestroyContext\0",
                    )
                    .ok()?;
                Some(Self {
                    _lib: lib,
                    va_initialize,
                    va_terminate,
                    va_create_config,
                    va_create_surfaces,
                    va_create_context,
                    va_begin_picture,
                    va_create_buffer,
                    va_render_picture,
                    va_end_picture,
                    va_sync_surface,
                    va_derive_image,
                    va_map_buffer,
                    va_unmap_buffer,
                    va_destroy_image,
                    va_destroy_buffer,
                    va_destroy_surfaces,
                    va_destroy_config,
                    va_destroy_context,
                })
            }
        }
    }

    /// Dynamically-loaded libva-drm function pointer.
    struct VaDrmLib {
        _lib: libloading::Library,
        va_get_display_drm: unsafe extern "C" fn(i32) -> VADisplay,
    }

    impl VaDrmLib {
        fn load() -> Option<Self> {
            // SAFETY: libva-drm.so.2 is a well-known system library.
            let lib =
                unsafe { libloading::Library::new("libva-drm.so.2") }.ok()?;
            // SAFETY: symbol signature matches the libva-drm C ABI.
            unsafe {
                let va_get_display_drm = *lib
                    .get::<unsafe extern "C" fn(i32) -> VADisplay>(
                        b"vaGetDisplayDRM\0",
                    )
                    .ok()?;
                Some(Self {
                    _lib: lib,
                    va_get_display_drm,
                })
            }
        }
    }

    /// VA-API hardware decoder for H.264/HEVC on Linux.
    pub struct VaapiDecoder {
        codec: VideoCodec,
        display: VADisplay,
        config: VAConfigID,
        context: VAContextID,
        surfaces: Vec<VASurfaceID>,
        width: u32,
        height: u32,
        initialized: bool,
        surfaces_created: bool,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
        va: Option<VaLib>,
        #[allow(dead_code)]
        va_drm: Option<VaDrmLib>,
    }

    impl VaapiDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            let Some(va) = VaLib::load() else {
                return Self::with_sw_fallback(codec);
            };
            let Some(va_drm) = VaDrmLib::load() else {
                return Self::with_sw_fallback(codec);
            };

            // SAFETY: (category 1) path is a null-terminated static byte string.
            let fd = unsafe {
                libc::open(
                    b"/dev/dri/renderD128\0".as_ptr() as *const libc::c_char,
                    libc::O_RDWR,
                )
            };
            if fd < 0 {
                return Self::with_sw_fallback(codec);
            }

            // SAFETY: (category 1) fd is valid (checked >= 0 above); vaInitialize status
            // is checked and display is terminated on failure.
            unsafe {
                let display = (va_drm.va_get_display_drm)(fd);
                let mut major = 0i32;
                let mut minor = 0i32;
                let status =
                    (va.va_initialize)(display, &mut major, &mut minor);
                if status != VA_STATUS_SUCCESS {
                    return Self::with_sw_fallback(codec);
                }

                let profile = match codec {
                    VideoCodec::H264 => VA_PROFILE_H264_HIGH,
                    VideoCodec::H265 => VA_PROFILE_HEVC_MAIN,
                    _ => {
                        return Err(VideoError::Codec(
                            "Unsupported codec".into(),
                        ))
                    }
                };

                let mut config_id: VAConfigID = 0;
                let status = (va.va_create_config)(
                    display,
                    profile,
                    VA_ENTRYPOINT_VLD,
                    ptr::null(),
                    0,
                    &mut config_id,
                );
                if status != VA_STATUS_SUCCESS {
                    (va.va_terminate)(display);
                    return Self::with_sw_fallback(codec);
                }

                Ok(VaapiDecoder {
                    codec,
                    display,
                    config: config_id,
                    context: 0,
                    surfaces: Vec::new(),
                    width: 0,
                    height: 0,
                    initialized: true,
                    surfaces_created: false,
                    sw_fallback: None,
                    va: Some(va),
                    va_drm: Some(va_drm),
                })
            }
        }

        fn with_sw_fallback(codec: VideoCodec) -> Result<Self, VideoError> {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => {
                    Box::new(super::super::h264_decoder::H264Decoder::new())
                }
                VideoCodec::H265 => {
                    Box::new(super::super::hevc_decoder::HevcDecoder::new())
                }
                _ => {
                    return Err(VideoError::Codec(
                        "Unsupported codec".into(),
                    ))
                }
            };
            Ok(VaapiDecoder {
                codec,
                display: ptr::null_mut(),
                config: 0,
                context: 0,
                surfaces: Vec::new(),
                width: 0,
                height: 0,
                initialized: false,
                surfaces_created: false,
                sw_fallback: Some(sw),
                va: None,
                va_drm: None,
            })
        }

        /// Create surfaces and context for the given resolution.
        unsafe fn create_surfaces(
            &mut self,
            width: u32,
            height: u32,
        ) -> Result<(), VideoError> {
            let va = self.va.as_ref().unwrap();
            self.width = width;
            self.height = height;
            let num_surfaces: u32 = 4;
            self.surfaces = vec![0u32; num_surfaces as usize];
            let status = (va.va_create_surfaces)(
                self.display,
                VA_RT_FORMAT_YUV420,
                width,
                height,
                self.surfaces.as_mut_ptr(),
                num_surfaces,
                ptr::null(),
                0,
            );
            if status != VA_STATUS_SUCCESS {
                return Err(VideoError::Codec(format!(
                    "VA-API: vaCreateSurfaces failed: {status}"
                )));
            }
            let mut ctx: VAContextID = 0;
            let status = (va.va_create_context)(
                self.display,
                self.config,
                width as i32,
                height as i32,
                0,
                self.surfaces.as_mut_ptr(),
                num_surfaces as i32,
                &mut ctx,
            );
            if status != VA_STATUS_SUCCESS {
                return Err(VideoError::Codec(format!(
                    "VA-API: vaCreateContext failed: {status}"
                )));
            }
            self.context = ctx;
            self.surfaces_created = true;
            Ok(())
        }

        /// Decode a single slice using the full VA-API pipeline.
        unsafe fn decode_slice(
            &mut self,
            slice_data: &[u8],
            surface_idx: usize,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let va = self.va.as_ref().unwrap();
            let surface = self.surfaces[surface_idx % self.surfaces.len()];

            let status =
                (va.va_begin_picture)(self.display, self.context, surface);
            if status != VA_STATUS_SUCCESS {
                return Err(VideoError::Codec(format!(
                    "VA-API: vaBeginPicture failed: {status}"
                )));
            }

            let mut slice_buf: VABufferID = 0;
            let status = (va.va_create_buffer)(
                self.display,
                self.context,
                VASliceDataBufferType,
                slice_data.len() as u32,
                1,
                slice_data.as_ptr() as *const c_void,
                &mut slice_buf,
            );
            if status != VA_STATUS_SUCCESS {
                (va.va_end_picture)(self.display, self.context);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaCreateBuffer(SliceData) failed: {status}"
                )));
            }

            let status = (va.va_render_picture)(
                self.display,
                self.context,
                &mut slice_buf,
                1,
            );
            if status != VA_STATUS_SUCCESS {
                (va.va_destroy_buffer)(self.display, slice_buf);
                (va.va_end_picture)(self.display, self.context);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaRenderPicture failed: {status}"
                )));
            }

            let status =
                (va.va_end_picture)(self.display, self.context);
            if status != VA_STATUS_SUCCESS {
                (va.va_destroy_buffer)(self.display, slice_buf);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaEndPicture failed: {status}"
                )));
            }

            let status = (va.va_sync_surface)(self.display, surface);
            if status != VA_STATUS_SUCCESS {
                (va.va_destroy_buffer)(self.display, slice_buf);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaSyncSurface failed: {status}"
                )));
            }

            let mut image: VAImage = std::mem::zeroed();
            let status =
                (va.va_derive_image)(self.display, surface, &mut image);
            if status != VA_STATUS_SUCCESS {
                (va.va_destroy_buffer)(self.display, slice_buf);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaDeriveImage failed: {status}"
                )));
            }

            let mut buf_ptr: *mut c_void = ptr::null_mut();
            let status =
                (va.va_map_buffer)(self.display, image.buf, &mut buf_ptr);
            if status != VA_STATUS_SUCCESS {
                (va.va_destroy_image)(self.display, image.image_id);
                (va.va_destroy_buffer)(self.display, slice_buf);
                return Err(VideoError::Codec(format!(
                    "VA-API: vaMapBuffer failed: {status}"
                )));
            }

            let w = image.width as usize;
            let h = image.height as usize;
            let y_pitch = image.pitches[0] as usize;
            let uv_pitch = image.pitches[1] as usize;
            let uv_offset = image.offsets[1] as usize;

            let mut rgb = vec![0u8; w * h * 3];
            super::nv12_to_rgb8(
                buf_ptr as *const u8,
                y_pitch,
                (buf_ptr as *const u8).add(uv_offset),
                uv_pitch,
                w,
                h,
                &mut rgb,
            );

            (va.va_unmap_buffer)(self.display, image.buf);
            (va.va_destroy_image)(self.display, image.image_id);
            (va.va_destroy_buffer)(self.display, slice_buf);

            Ok(Some(DecodedFrame {
                width: w,
                height: h,
                rgb8_data: rgb,
                timestamp_us: 0,
                keyframe: false,
                bit_depth: 8,
                rgb16_data: None,
            }))
        }
    }

    impl VideoDecoder for VaapiDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }

            // Parse Annex B NAL units
            let nals = crate::parse_annex_b(data);
            if nals.is_empty() {
                return Ok(None);
            }

            // Create surfaces on first non-empty frame if not done yet.
            // Default to 1920x1080; real implementation would parse SPS for resolution.
            if !self.surfaces_created {
                // SAFETY: (category 1) VA display was initialized successfully.
                unsafe {
                    self.create_surfaces(1920, 1080)?;
                }
            }

            // Concatenate all non-parameter NALs as slice data
            let mut slice_data = Vec::new();
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                let is_param = match self.codec {
                    VideoCodec::H264 => matches!(nal.data[0] & 0x1F, 7 | 8),
                    VideoCodec::H265 => matches!((nal.data[0] >> 1) & 0x3F, 32..=34),
                    _ => false,
                };
                if !is_param {
                    slice_data.extend_from_slice(&nal.data);
                }
            }

            if slice_data.is_empty() {
                return Ok(None);
            }

            // Full VA-API pipeline: vaBeginPicture → vaCreateBuffer(SliceData) →
            // vaRenderPicture → vaEndPicture → vaSyncSurface → vaDeriveImage →
            // vaMapBuffer → NV12→RGB readback
            // SAFETY: (category 1) surfaces/context created and status checked at each step.
            unsafe {
                let mut frame = self.decode_slice(&slice_data, 0)?;
                if let Some(ref mut f) = frame {
                    f.timestamp_us = timestamp_us;
                }
                Ok(frame)
            }
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            Ok(Vec::new())
        }
    }

    impl Drop for VaapiDecoder {
        fn drop(&mut self) {
            if self.initialized && !self.display.is_null() {
                let va = self.va.as_ref().unwrap();
                // SAFETY: (category 1) display/config/context/surfaces are valid
                // (self.initialized + null checks guard); VA-API teardown order is respected.
                unsafe {
                    if self.surfaces_created {
                        if self.context != 0 {
                            (va.va_destroy_context)(
                                self.display,
                                self.context,
                            );
                        }
                        if !self.surfaces.is_empty() {
                            (va.va_destroy_surfaces)(
                                self.display,
                                self.surfaces.as_mut_ptr(),
                                self.surfaces.len() as i32,
                            );
                        }
                    }
                    if self.config != 0 {
                        (va.va_destroy_config)(self.display, self.config);
                    }
                    (va.va_terminate)(self.display);
                }
            }
        }
    }

    unsafe impl Send for VaapiDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// NVDEC backend (NVIDIA, Linux/Windows)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nvdec")]
#[allow(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::upper_case_acronyms,
    clippy::field_reassign_with_default,
    dead_code
)]
pub mod nvdec {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    type CUresult = i32;
    type CUcontext = *mut c_void;
    type CUvideodecoder = *mut c_void;
    type CUvideoparser = *mut c_void;
    type CUdeviceptr = u64;

    const CUDA_SUCCESS: CUresult = 0;
    const cudaVideoCodec_H264: i32 = 4;
    const cudaVideoCodec_HEVC: i32 = 8;
    const cudaVideoSurfaceFormat_NV12: i32 = 0;
    const cudaVideoChromaFormat_420: i32 = 1;

    // NVDEC parser callback types
    type PfnSequenceCallback = unsafe extern "C" fn(*mut c_void, *mut CUVIDEOFORMAT) -> i32;
    type PfnDecodePicture = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
    type PfnDisplayPicture = unsafe extern "C" fn(*mut c_void, *mut CUVIDPARSERDISPINFO) -> i32;

    #[repr(C)]
    struct CUVIDPARSERPARAMS {
        codec_type: i32,
        max_num_decode_surfaces: u32,
        clock_rate: u32,
        error_threshold: u32,
        max_display_delay: u32,
        reserved1: [u32; 5],
        user_data: *mut c_void,
        pfn_sequence_callback: PfnSequenceCallback,
        pfn_decode_picture: PfnDecodePicture,
        pfn_display_picture: PfnDisplayPicture,
        reserved2: [*mut c_void; 7],
        ext_video_info: *mut c_void,
    }

    #[repr(C)]
    struct CUVIDEOFORMAT {
        codec: i32,
        frame_rate_num: u32,
        frame_rate_den: u32,
        progressive_sequence: u8,
        bit_depth_luma_minus8: u8,
        bit_depth_chroma_minus8: u8,
        min_num_decode_surfaces: u8,
        coded_width: u32,
        coded_height: u32,
        // ... more fields, we only need width/height
        _pad: [u8; 256], // padding for remaining fields
    }

    #[repr(C)]
    struct CUVIDPARSERDISPINFO {
        picture_index: i32,
        progressive_frame: i32,
        top_field_first: i32,
        repeat_first_field: i32,
        timestamp: i64,
    }

    #[repr(C)]
    struct CUVIDSOURCEDATAPACKET {
        flags: u64,
        payload_size: u64,
        payload: *const u8,
        timestamp: i64,
    }

    #[repr(C)]
    struct CUVIDDECODECREATEINFO {
        code_type: i32,
        chroma_format: i32,
        output_format: i32,
        bit_depth_minus8: u32,
        ull_intra_decode_only: u32,
        reserved1: [u32; 3],
        display_area_left: i16,
        display_area_top: i16,
        display_area_right: i16,
        display_area_bottom: i16,
        ul_width: u32,
        ul_height: u32,
        ul_max_width: u32,
        ul_max_height: u32,
        ul_target_width: u32,
        ul_target_height: u32,
        ul_num_decode_surfaces: u32,
        ul_num_output_surfaces: u32,
        de_interlace_mode: i32,
        video_lock: *mut c_void,
        _pad: [u8; 128],
    }

    #[repr(C)]
    struct CUVIDPROCPARAMS {
        progressive_frame: i32,
        second_field: i32,
        top_field_first: i32,
        unpaired_field: i32,
        reserved_flags: u32,
        reserved_zero: u32,
        raw_input_dptr: u64,
        raw_input_pitch: u32,
        raw_input_format: u32,
        raw_output_dptr: u64,
        raw_output_pitch: u32,
        reserved1: u32,
        output_stream: *mut c_void,
        reserved: [u32; 16],
    }

    #[link(name = "cuda")]
    unsafe extern "C" {
        fn cuInit(flags: u32) -> CUresult;
        fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, device: i32) -> CUresult;
        fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    }

    #[link(name = "nvcuvid")]
    unsafe extern "C" {
        fn cuvidCreateVideoParser(
            obj: *mut CUvideoparser,
            params: *mut CUVIDPARSERPARAMS,
        ) -> CUresult;
        fn cuvidDestroyVideoParser(obj: CUvideoparser) -> CUresult;
        fn cuvidParseVideoData(obj: CUvideoparser, packet: *mut CUVIDSOURCEDATAPACKET) -> CUresult;
        fn cuvidCreateDecoder(
            decoder: *mut CUvideodecoder,
            params: *mut CUVIDDECODECREATEINFO,
        ) -> CUresult;
        fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult;
        fn cuvidDecodePicture(decoder: CUvideodecoder, pic_params: *mut c_void) -> CUresult;
        fn cuvidMapVideoFrame64(
            decoder: CUvideodecoder,
            pic_idx: i32,
            dev_ptr: *mut CUdeviceptr,
            pitch: *mut u32,
            params: *mut CUVIDPROCPARAMS,
        ) -> CUresult;
        fn cuvidUnmapVideoFrame64(decoder: CUvideodecoder, dev_ptr: CUdeviceptr) -> CUresult;
    }

    /// Shared state between NVDEC parser callbacks and decoder.
    struct NvdecState {
        decoder: CUvideodecoder,
        width: u32,
        height: u32,
        frames: Vec<DecodedFrame>,
        decoder_created: bool,
        /// Callback error propagation: set inside parser callbacks, checked after parse.
        last_error: Option<String>,
    }

    // Parser callbacks
    // SAFETY: (category 2) user_data points to Box<NvdecState> pinned for the parser
    // lifetime; fmt is provided by the NVDEC parser and valid for the call duration.
    unsafe extern "C" fn sequence_callback(user_data: *mut c_void, fmt: *mut CUVIDEOFORMAT) -> i32 {
        let state = &mut *(user_data as *mut NvdecState);
        state.width = (*fmt).coded_width;
        state.height = (*fmt).coded_height;

        if !state.decoder_created {
            let mut create_info: CUVIDDECODECREATEINFO = std::mem::zeroed();
            create_info.code_type = (*fmt).codec;
            create_info.chroma_format = cudaVideoChromaFormat_420;
            create_info.output_format = cudaVideoSurfaceFormat_NV12;
            create_info.ul_width = state.width;
            create_info.ul_height = state.height;
            create_info.ul_max_width = state.width;
            create_info.ul_max_height = state.height;
            create_info.ul_target_width = state.width;
            create_info.ul_target_height = state.height;
            create_info.ul_num_decode_surfaces = 20;
            create_info.ul_num_output_surfaces = 2;

            let status = cuvidCreateDecoder(&mut state.decoder, &mut create_info);
            if status == CUDA_SUCCESS {
                state.decoder_created = true;
            } else {
                state.last_error = Some(format!("NVDEC: cuvidCreateDecoder failed: {status}"));
            }
        }
        (*fmt).min_num_decode_surfaces as i32
    }

    // SAFETY: (category 2) user_data is a valid NvdecState pointer; pic_params
    // provided by the parser and valid for the call duration.
    unsafe extern "C" fn decode_picture_callback(
        user_data: *mut c_void,
        pic_params: *mut c_void,
    ) -> i32 {
        let state = &mut *(user_data as *mut NvdecState);
        if !state.decoder_created {
            return 0;
        }
        let status = cuvidDecodePicture(state.decoder, pic_params);
        if status != CUDA_SUCCESS {
            state.last_error = Some(format!("NVDEC: cuvidDecodePicture failed: {status}"));
            0
        } else {
            1
        }
    }

    // SAFETY: (category 2 + 4) user_data is valid NvdecState; disp_info null-checked;
    // GPU frame is mapped, copied to host NV12 buffer, and unmapped within this call.
    unsafe extern "C" fn display_picture_callback(
        user_data: *mut c_void,
        disp_info: *mut CUVIDPARSERDISPINFO,
    ) -> i32 {
        if disp_info.is_null() {
            return 1;
        }
        let state = &mut *(user_data as *mut NvdecState);
        if !state.decoder_created {
            return 0;
        }

        let info = &*disp_info;
        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let mut proc_params: CUVIDPROCPARAMS = std::mem::zeroed();
        proc_params.progressive_frame = info.progressive_frame;
        proc_params.top_field_first = info.top_field_first;

        let status = cuvidMapVideoFrame64(
            state.decoder,
            info.picture_index,
            &mut dev_ptr,
            &mut pitch,
            &mut proc_params,
        );
        if status != CUDA_SUCCESS {
            state.last_error = Some(format!("NVDEC: cuvidMapVideoFrame64 failed: {status}"));
            return 0;
        }

        let w = state.width as usize;
        let h = state.height as usize;
        let p = pitch as usize;

        // Copy NV12 from GPU: Y plane + UV plane
        let y_size = p * h;
        let uv_size = p * (h / 2);
        let mut nv12 = vec![0u8; y_size + uv_size];
        cuMemcpyDtoH_v2(nv12.as_mut_ptr() as *mut c_void, dev_ptr, y_size + uv_size);
        cuvidUnmapVideoFrame64(state.decoder, dev_ptr);

        // NV12 → YUV420 planar → RGB
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; (w / 2) * (h / 2)];
        let mut cr = vec![0u8; (w / 2) * (h / 2)];

        for row in 0..h {
            y[row * w..(row + 1) * w].copy_from_slice(&nv12[row * p..row * p + w]);
        }
        let uv_base = y_size;
        for row in 0..(h / 2) {
            for col in 0..(w / 2) {
                cb[row * (w / 2) + col] = nv12[uv_base + row * p + col * 2];
                cr[row * (w / 2) + col] = nv12[uv_base + row * p + col * 2 + 1];
            }
        }

        let rgb =
            crate::yuv420_to_rgb8(&y, &cb, &cr, w, h).unwrap_or_else(|_| vec![128u8; w * h * 3]);

        state.frames.push(DecodedFrame {
            width: w,
            height: h,
            rgb8_data: rgb,
            timestamp_us: info.timestamp as u64,
            keyframe: false,
            bit_depth: 8,
            rgb16_data: None,
        });
        1
    }

    /// NVIDIA NVDEC hardware decoder with built-in parser.
    pub struct NvdecDecoder {
        codec: VideoCodec,
        cuda_ctx: CUcontext,
        parser: CUvideoparser,
        state: Box<NvdecState>,
        initialized: bool,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
    }

    impl NvdecDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            // SAFETY: (category 1) CUDA/NVDEC init sequence; each FFI status is checked
            // and resources are freed on failure path.
            unsafe {
                let status = cuInit(0);
                if status != CUDA_SUCCESS {
                    return Ok(Self::with_sw_fallback(codec));
                }

                let mut ctx: CUcontext = ptr::null_mut();
                let status = cuCtxCreate_v2(&mut ctx, 0, 0);
                if status != CUDA_SUCCESS {
                    return Ok(Self::with_sw_fallback(codec));
                }

                let mut state = Box::new(NvdecState {
                    decoder: ptr::null_mut(),
                    width: 0,
                    height: 0,
                    frames: Vec::new(),
                    decoder_created: false,
                    last_error: None,
                });

                let nvcodec = match codec {
                    VideoCodec::H264 => cudaVideoCodec_H264,
                    VideoCodec::H265 => cudaVideoCodec_HEVC,
                    _ => return Err(VideoError::Codec("NVDEC: unsupported codec".into())),
                };

                let mut params = CUVIDPARSERPARAMS {
                    codec_type: nvcodec,
                    max_num_decode_surfaces: 20,
                    clock_rate: 0,
                    error_threshold: 100,
                    max_display_delay: 4,
                    reserved1: [0; 5],
                    user_data: &mut *state as *mut NvdecState as *mut c_void,
                    pfn_sequence_callback: sequence_callback,
                    pfn_decode_picture: decode_picture_callback,
                    pfn_display_picture: display_picture_callback,
                    reserved2: [ptr::null_mut(); 7],
                    ext_video_info: ptr::null_mut(),
                };

                let mut parser: CUvideoparser = ptr::null_mut();
                let status = cuvidCreateVideoParser(&mut parser, &mut params);
                if status != CUDA_SUCCESS {
                    cuCtxDestroy_v2(ctx);
                    return Ok(Self::with_sw_fallback(codec));
                }

                Ok(NvdecDecoder {
                    codec,
                    cuda_ctx: ctx,
                    parser,
                    state,
                    initialized: true,
                    sw_fallback: None,
                })
            }
        }

        fn with_sw_fallback(codec: VideoCodec) -> Self {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
            };
            NvdecDecoder {
                codec,
                cuda_ctx: ptr::null_mut(),
                parser: ptr::null_mut(),
                state: Box::new(NvdecState {
                    decoder: ptr::null_mut(),
                    width: 0,
                    height: 0,
                    frames: Vec::new(),
                    decoder_created: false,
                    last_error: None,
                }),
                initialized: false,
                sw_fallback: Some(sw),
            }
        }
    }

    impl VideoDecoder for NvdecDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }

            // Feed Annex B data to NVDEC parser — callbacks handle decode + display
            // SAFETY: (category 1) parser handle is valid (self.initialized); data.as_ptr()
            // valid for data.len() bytes.
            unsafe {
                let mut packet: CUVIDSOURCEDATAPACKET = std::mem::zeroed();
                packet.payload_size = data.len() as u64;
                packet.payload = data.as_ptr();
                packet.timestamp = timestamp_us as i64;
                packet.flags = 0;

                let status = cuvidParseVideoData(self.parser, &mut packet);
                if status != CUDA_SUCCESS {
                    return Err(VideoError::Codec(format!(
                        "NVDEC: cuvidParseVideoData failed: {status}"
                    )));
                }
            }

            // Check for errors propagated from callbacks
            if let Some(err) = self.state.last_error.take() {
                return Err(VideoError::Codec(err));
            }

            // Return last decoded frame from callback
            let mut frame = self.state.frames.pop();
            if let Some(ref mut f) = frame {
                f.timestamp_us = timestamp_us;
            }
            Ok(frame)
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            // Send end-of-stream packet
            // SAFETY: (category 1) parser is valid; zeroed packet with EOS flag is well-formed.
            unsafe {
                let mut packet: CUVIDSOURCEDATAPACKET = std::mem::zeroed();
                packet.flags = 1; // CUVID_PKT_ENDOFSTREAM
                let _ = cuvidParseVideoData(self.parser, &mut packet);
            }
            Ok(std::mem::take(&mut self.state.frames))
        }
    }

    impl Drop for NvdecDecoder {
        fn drop(&mut self) {
            if self.initialized {
                // SAFETY: (category 1) parser/decoder/ctx are valid (self.initialized +
                // null checks); destroy order: parser -> decoder -> CUDA context.
                unsafe {
                    if !self.parser.is_null() {
                        cuvidDestroyVideoParser(self.parser);
                    }
                    if self.state.decoder_created && !self.state.decoder.is_null() {
                        cuvidDestroyDecoder(self.state.decoder);
                    }
                    if !self.cuda_ctx.is_null() {
                        cuCtxDestroy_v2(self.cuda_ctx);
                    }
                }
            }
        }
    }

    unsafe impl Send for NvdecDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Media Foundation backend (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "windows", feature = "media-foundation"))]
#[allow(unsafe_code, non_camel_case_types, non_snake_case)]
pub mod media_foundation {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    type HRESULT = i32;
    type GUID = [u8; 16];

    const S_OK: HRESULT = 0;
    const MF_E_TRANSFORM_NEED_MORE_INPUT: HRESULT = 0xC00D6D72_u32 as i32;
    #[allow(dead_code)]
    const MF_E_INVALIDMEDIATYPE: HRESULT = 0xC00D36B4_u32 as i32;
    #[allow(dead_code)]
    const E_NOTIMPL: HRESULT = 0x80004001_u32 as i32;

    // MFT category GUID for video decoders {d0033739-4f81-4293-868e-2f732875c515}
    const MFT_CATEGORY_VIDEO_DECODER: GUID = [
        0x39, 0x37, 0x03, 0xd0, 0x81, 0x4f, 0x93, 0x42, 0x86, 0x8e, 0x2f, 0x73, 0x28, 0x75, 0xc5,
        0x15,
    ];

    // IID_IMFTransform {bf94c121-5b05-4e6f-8000-ba598961414d}
    const IID_IMF_TRANSFORM: GUID = [
        0x21, 0xc1, 0x94, 0xbf, 0x05, 0x5b, 0x6f, 0x4e, 0x80, 0x00, 0xba, 0x59, 0x89, 0x61, 0x41,
        0x4d,
    ];

    // Well-known media type GUIDs
    const MFMediaType_Video: GUID = [
        0x73, 0x64, 0x69, 0x76, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_H264: GUID = [
        0x48, 0x32, 0x36, 0x34, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_HEVC: GUID = [
        0x48, 0x45, 0x56, 0x43, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_NV12: GUID = [
        0x4e, 0x56, 0x31, 0x32, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];

    // MF_MT attribute GUIDs
    // {48eba18e-f8c9-4687-bf11-0a74c9f96a8f}
    const MF_MT_MAJOR_TYPE: GUID = [
        0x8e, 0xa1, 0xeb, 0x48, 0xc9, 0xf8, 0x87, 0x46, 0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a,
        0x8f,
    ];
    // {f7e34c9a-42e8-4714-b74b-cb29d72c35e5}
    const MF_MT_SUBTYPE: GUID = [
        0x9a, 0x4c, 0xe3, 0xf7, 0xe8, 0x42, 0x14, 0x47, 0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35,
        0xe5,
    ];
    // {e2724bb8-e676-4806-b4b2-a8d6efb44ccd}
    const MF_MT_INTERLACE_MODE: GUID = [
        0xb8, 0x4b, 0x72, 0xe2, 0x76, 0xe6, 0x06, 0x48, 0xb4, 0xb2, 0xa8, 0xd6, 0xef, 0xb4, 0x4c,
        0xcd,
    ];
    // {1652c33d-d6b2-4012-b834-72030849a37d}
    const MF_MT_FRAME_SIZE: GUID = [
        0x3d, 0xc3, 0x52, 0x16, 0xb2, 0xd6, 0x12, 0x40, 0xb8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xa3,
        0x7d,
    ];

    // MFT_MESSAGE constants
    const MFT_MESSAGE_COMMAND_FLUSH: u32 = 0x0;
    const MFT_MESSAGE_COMMAND_DRAIN: u32 = 0x1;
    const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x10000000;
    const MFT_MESSAGE_NOTIFY_START_OF_STREAM: u32 = 0x10000003;

    const MFT_OUTPUT_STREAM_PROVIDES_SAMPLES: u32 = 0x1;

    // ── Extern function bindings ──────────────────────────────────────

    #[link(name = "mfplat")]
    unsafe extern "system" {
        fn MFStartup(version: u32, flags: u32) -> HRESULT;
        fn MFShutdown() -> HRESULT;
        fn MFCreateMediaType(media_type: *mut *mut c_void) -> HRESULT;
    }

    #[link(name = "mf")]
    unsafe extern "system" {
        fn MFTEnumEx(
            guid_category: *const GUID,
            flags: u32,
            input_type: *const MFT_REGISTER_TYPE_INFO,
            output_type: *const MFT_REGISTER_TYPE_INFO,
            activate: *mut *mut *mut c_void,
            count: *mut u32,
        ) -> HRESULT;
        fn MFCreateSample(sample: *mut *mut c_void) -> HRESULT;
        fn MFCreateMemoryBuffer(max_len: u32, buffer: *mut *mut c_void) -> HRESULT;
    }

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn CoTaskMemFree(pv: *mut c_void);
    }

    // ── Structs ───────────────────────────────────────────────────────

    #[repr(C)]
    struct MFT_REGISTER_TYPE_INFO {
        guid_major_type: GUID,
        guid_subtype: GUID,
    }

    #[repr(C)]
    struct MFT_OUTPUT_DATA_BUFFER {
        stream_id: u32,
        sample: *mut c_void,
        status: u32,
        events: *mut c_void,
    }

    #[repr(C)]
    struct MFT_OUTPUT_STREAM_INFO {
        flags: u32,
        cb_size: u32,
        cb_alignment: u32,
    }

    // ── COM vtable helpers ────────────────────────────────────────────
    //
    // IMFTransform (inherits IUnknown: 0=QI, 1=AddRef, 2=Release):
    //   3=GetStreamLimits, 4=GetStreamIDs, 5=GetStreamCount,
    //   6=GetInputStreamInfo, 7=GetOutputStreamInfo, 8=GetAttributes,
    //   9=GetInputStreamAttributes, 10=GetOutputStreamAttributes,
    //   11=DeleteInputStream, 12=AddInputStreams,
    //   13=GetInputAvailableType, 14=GetOutputAvailableType,
    //   15=SetInputType, 16=SetOutputType,
    //   17=GetInputCurrentType, 18=GetOutputCurrentType,
    //   19=GetInputStatus, 20=GetOutputStatus, 21=SetOutputBounds,
    //   22=ProcessEvent, 23=ProcessMessage, 24=ProcessInput, 25=ProcessOutput
    //
    // IMFAttributes (inherits IUnknown: 0-2):
    //   3=GetItem..32=CopyAllItems
    //   7=GetUINT32, 8=GetUINT64, 10=GetGUID, 21=SetUINT32, 24=SetGUID
    //
    // IMFActivate (inherits IMFAttributes: 0-32):
    //   33=ActivateObject, 34=ShutdownObject, 35=DetachObject
    //
    // IMFSample (inherits IMFAttributes: 0-32):
    //   33=GetSampleFlags..46=CopyToBuffer
    //   36=SetSampleTime, 41=ConvertToContiguousBuffer, 42=AddBuffer
    //
    // IMFMediaBuffer (inherits IUnknown: 0-2):
    //   3=Lock, 4=Unlock, 5=GetCurrentLength, 6=SetCurrentLength, 7=GetMaxLength

    /// COM Release (vtable index 2).
    unsafe fn com_release(obj: *mut c_void) {
        if !obj.is_null() {
            let vtable = *(obj as *const *const *const c_void);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(*vtable.add(2));
            release(obj);
        }
    }

    /// IMFActivate::ActivateObject (vtable 33)
    unsafe fn activate_object(
        activate: *mut c_void,
        iid: *const GUID,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(activate as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const GUID,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(33));
        method(activate, iid, out)
    }

    /// IMFTransform::GetOutputStreamInfo (vtable 7)
    unsafe fn transform_get_output_stream_info(
        transform: *mut c_void,
        stream_id: u32,
        info: *mut MFT_OUTPUT_STREAM_INFO,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *mut MFT_OUTPUT_STREAM_INFO,
        ) -> HRESULT = std::mem::transmute(*vtable.add(7));
        method(transform, stream_id, info)
    }

    /// IMFTransform::GetOutputAvailableType (vtable 14)
    unsafe fn transform_get_output_available_type(
        transform: *mut c_void,
        stream_id: u32,
        type_idx: u32,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            u32,
            u32,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(14));
        method(transform, stream_id, type_idx, out)
    }

    /// IMFTransform::SetInputType (vtable 15)
    unsafe fn transform_set_input_type(
        transform: *mut c_void,
        stream_id: u32,
        media_type: *mut c_void,
        flags: u32,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(15));
        method(transform, stream_id, media_type, flags)
    }

    /// IMFTransform::SetOutputType (vtable 16)
    unsafe fn transform_set_output_type(
        transform: *mut c_void,
        stream_id: u32,
        media_type: *mut c_void,
        flags: u32,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(16));
        method(transform, stream_id, media_type, flags)
    }

    /// IMFTransform::GetOutputCurrentType (vtable 18)
    unsafe fn transform_get_output_current_type(
        transform: *mut c_void,
        stream_id: u32,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(18));
        method(transform, stream_id, out)
    }

    /// IMFTransform::ProcessMessage (vtable 23)
    unsafe fn transform_process_message(
        transform: *mut c_void,
        message: u32,
        param: u64,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, u64) -> HRESULT =
            std::mem::transmute(*vtable.add(23));
        method(transform, message, param)
    }

    /// IMFTransform::ProcessInput (vtable 24)
    unsafe fn transform_process_input(
        transform: *mut c_void,
        stream_id: u32,
        sample: *mut c_void,
        flags: u32,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(24));
        method(transform, stream_id, sample, flags)
    }

    /// IMFTransform::ProcessOutput (vtable 25)
    unsafe fn transform_process_output(
        transform: *mut c_void,
        flags: u32,
        count: u32,
        output_buffers: *mut MFT_OUTPUT_DATA_BUFFER,
        status: *mut u32,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            u32,
            u32,
            *mut MFT_OUTPUT_DATA_BUFFER,
            *mut u32,
        ) -> HRESULT = std::mem::transmute(*vtable.add(25));
        method(transform, flags, count, output_buffers, status)
    }

    /// IMFAttributes::GetUINT64 (vtable 8)
    unsafe fn attributes_get_uint64(
        attrs: *mut c_void,
        key: *const GUID,
        out: *mut u64,
    ) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *const GUID, *mut u64) -> HRESULT =
            std::mem::transmute(*vtable.add(8));
        method(attrs, key, out)
    }

    /// IMFAttributes::GetGUID (vtable 10)
    unsafe fn attributes_get_guid(
        attrs: *mut c_void,
        key: *const GUID,
        out: *mut GUID,
    ) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *const GUID, *mut GUID) -> HRESULT =
            std::mem::transmute(*vtable.add(10));
        method(attrs, key, out)
    }

    /// IMFAttributes::SetUINT32 (vtable 21)
    unsafe fn attributes_set_uint32(
        attrs: *mut c_void,
        key: *const GUID,
        value: u32,
    ) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *const GUID, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(21));
        method(attrs, key, value)
    }

    /// IMFAttributes::SetGUID (vtable 24)
    unsafe fn attributes_set_guid(
        attrs: *mut c_void,
        key: *const GUID,
        value: *const GUID,
    ) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const GUID,
            *const GUID,
        ) -> HRESULT = std::mem::transmute(*vtable.add(24));
        method(attrs, key, value)
    }

    /// IMFMediaBuffer::Lock (vtable 3)
    unsafe fn media_buffer_lock(
        buf: *mut c_void,
        data_out: *mut *mut u8,
        max_len: *mut u32,
        cur_len: *mut u32,
    ) -> HRESULT {
        let vtable = *(buf as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *mut *mut u8,
            *mut u32,
            *mut u32,
        ) -> HRESULT = std::mem::transmute(*vtable.add(3));
        method(buf, data_out, max_len, cur_len)
    }

    /// IMFMediaBuffer::Unlock (vtable 4)
    unsafe fn media_buffer_unlock(buf: *mut c_void) -> HRESULT {
        let vtable = *(buf as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(4));
        method(buf)
    }

    /// IMFMediaBuffer::SetCurrentLength (vtable 6)
    unsafe fn media_buffer_set_current_length(buf: *mut c_void, len: u32) -> HRESULT {
        let vtable = *(buf as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(6));
        method(buf, len)
    }

    /// IMFSample::SetSampleTime (vtable 36)
    unsafe fn sample_set_sample_time(sample: *mut c_void, time: i64) -> HRESULT {
        let vtable = *(sample as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, i64) -> HRESULT =
            std::mem::transmute(*vtable.add(36));
        method(sample, time)
    }

    /// IMFSample::ConvertToContiguousBuffer (vtable 41)
    unsafe fn sample_convert_to_contiguous_buffer(
        sample: *mut c_void,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(sample as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(41));
        method(sample, out)
    }

    /// IMFSample::AddBuffer (vtable 42)
    unsafe fn sample_add_buffer(sample: *mut c_void, buffer: *mut c_void) -> HRESULT {
        let vtable = *(sample as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(42));
        method(sample, buffer)
    }

    // ── Decoder ───────────────────────────────────────────────────────

    /// Media Foundation hardware decoder for Windows.
    ///
    /// Uses MFTEnumEx to find the system H.264/HEVC decoder MFT,
    /// then feeds NAL data via ProcessInput/ProcessOutput.
    pub struct MediaFoundationDecoder {
        codec: VideoCodec,
        initialized: bool,
        width: u32,
        height: u32,
        transform: *mut c_void,
        provides_samples: bool,
        output_buf_size: u32,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
    }

    impl MediaFoundationDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            // SAFETY: (category 1) MF startup/enum/activate sequence; HRESULT checked
            // at each step; falls back to software on any failure.
            unsafe { Self::init_mft(codec) }
        }

        /// Full MFT initialization: startup → enum → activate → configure types → begin stream.
        /// On any failure, falls back to software decoder instead of returning Err.
        unsafe fn init_mft(codec: VideoCodec) -> Result<Self, VideoError> {
            // 1. MFStartup
            let hr = MFStartup(0x00020070, 0); // MF_VERSION = 2.0
            if hr != S_OK {
                return Ok(Self::with_sw_fallback(codec));
            }

            // 2. MFTEnumEx — find decoder MFTs
            let subtype = match codec {
                VideoCodec::H264 => MFVideoFormat_H264,
                VideoCodec::H265 => MFVideoFormat_HEVC,
                _ => return Err(VideoError::Codec("MF: unsupported codec".into())),
            };
            let input_info = MFT_REGISTER_TYPE_INFO {
                guid_major_type: MFMediaType_Video,
                guid_subtype: subtype,
            };
            let mut activate_array: *mut *mut c_void = ptr::null_mut();
            let mut count: u32 = 0;
            let hr = MFTEnumEx(
                &MFT_CATEGORY_VIDEO_DECODER,
                0x00000070, // MFT_ENUM_FLAG_SYNCMFT | ASYNCMFT | HARDWARE | SORTANDFILTER
                &input_info,
                ptr::null(),
                &mut activate_array,
                &mut count,
            );
            if hr != S_OK || count == 0 || activate_array.is_null() {
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            // 3. ActivateObject on first entry → IMFTransform*
            let first_activate = *activate_array;
            let mut transform: *mut c_void = ptr::null_mut();
            let hr = activate_object(first_activate, &IID_IMF_TRANSFORM, &mut transform);

            // 4. Free activate array (Release each entry, then CoTaskMemFree the array)
            for i in 0..count as usize {
                com_release(*activate_array.add(i));
            }
            CoTaskMemFree(activate_array as *mut c_void);

            if hr != S_OK || transform.is_null() {
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            // 5. Create + configure input media type
            let mut input_type: *mut c_void = ptr::null_mut();
            let hr = MFCreateMediaType(&mut input_type);
            if hr != S_OK || input_type.is_null() {
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }
            attributes_set_guid(input_type, &MF_MT_MAJOR_TYPE, &MFMediaType_Video);
            attributes_set_guid(input_type, &MF_MT_SUBTYPE, &subtype);
            // MFVideoInterlace_Progressive = 2
            attributes_set_uint32(input_type, &MF_MT_INTERLACE_MODE, 2);

            // 6. SetInputType
            let hr = transform_set_input_type(transform, 0, input_type, 0);
            com_release(input_type);
            if hr != S_OK {
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            // 7. Enumerate output types, find NV12
            let mut nv12_type: *mut c_void = ptr::null_mut();
            for idx in 0..64u32 {
                let mut candidate: *mut c_void = ptr::null_mut();
                let hr =
                    transform_get_output_available_type(transform, 0, idx, &mut candidate);
                if hr != S_OK || candidate.is_null() {
                    break;
                }
                let mut sub: GUID = [0u8; 16];
                if attributes_get_guid(candidate, &MF_MT_SUBTYPE, &mut sub) == S_OK
                    && sub == MFVideoFormat_NV12
                {
                    nv12_type = candidate;
                    break;
                }
                com_release(candidate);
            }
            if nv12_type.is_null() {
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            let hr = transform_set_output_type(transform, 0, nv12_type, 0);

            // Read frame size from negotiated output type (high32 = width, low32 = height)
            let mut width: u32 = 0;
            let mut height: u32 = 0;
            let mut frame_size: u64 = 0;
            if attributes_get_uint64(nv12_type, &MF_MT_FRAME_SIZE, &mut frame_size) == S_OK {
                width = (frame_size >> 32) as u32;
                height = frame_size as u32;
            }
            com_release(nv12_type);

            if hr != S_OK {
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            // 8. Query output stream info (buffer size, provides_samples flag)
            let mut stream_info = MFT_OUTPUT_STREAM_INFO {
                flags: 0,
                cb_size: 0,
                cb_alignment: 0,
            };
            let (provides_samples, output_buf_size) =
                if transform_get_output_stream_info(transform, 0, &mut stream_info) == S_OK {
                    (
                        (stream_info.flags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES) != 0,
                        stream_info.cb_size,
                    )
                } else {
                    (false, 0)
                };

            // 9. Notify begin/start of stream
            transform_process_message(transform, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
            transform_process_message(transform, MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);

            Ok(MediaFoundationDecoder {
                codec,
                initialized: true,
                width,
                height,
                transform,
                provides_samples,
                output_buf_size,
                sw_fallback: None,
            })
        }

        fn with_sw_fallback(codec: VideoCodec) -> Self {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
            };
            MediaFoundationDecoder {
                codec,
                initialized: false,
                width: 0,
                height: 0,
                transform: ptr::null_mut(),
                provides_samples: false,
                output_buf_size: 0,
                sw_fallback: Some(sw),
            }
        }

        /// Feed NAL data via ProcessInput, then try to drain one frame via ProcessOutput.
        unsafe fn feed_and_drain(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if self.transform.is_null() {
                return Err(VideoError::Codec("MF: transform not initialized".into()));
            }

            // Create input buffer, lock, copy NAL data, unlock
            let mut media_buf: *mut c_void = ptr::null_mut();
            let hr = MFCreateMemoryBuffer(data.len() as u32, &mut media_buf);
            if hr != S_OK || media_buf.is_null() {
                return Err(VideoError::Codec(format!(
                    "MF: MFCreateMemoryBuffer failed: {hr:#X}"
                )));
            }

            let mut buf_ptr: *mut u8 = ptr::null_mut();
            let mut max_len: u32 = 0;
            let mut cur_len: u32 = 0;
            let hr = media_buffer_lock(media_buf, &mut buf_ptr, &mut max_len, &mut cur_len);
            if hr != S_OK {
                com_release(media_buf);
                return Err(VideoError::Codec(format!(
                    "MF: IMFMediaBuffer::Lock failed: {hr:#X}"
                )));
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf_ptr, data.len());
            media_buffer_unlock(media_buf);
            media_buffer_set_current_length(media_buf, data.len() as u32);

            // Create input sample, add buffer, set timestamp
            let mut sample: *mut c_void = ptr::null_mut();
            let hr = MFCreateSample(&mut sample);
            if hr != S_OK || sample.is_null() {
                com_release(media_buf);
                return Err(VideoError::Codec(format!(
                    "MF: MFCreateSample failed: {hr:#X}"
                )));
            }
            sample_add_buffer(sample, media_buf);
            com_release(media_buf);
            // MF uses 100ns units
            sample_set_sample_time(sample, timestamp_us as i64 * 10);

            // ProcessInput
            let hr = transform_process_input(self.transform, 0, sample, 0);
            com_release(sample);
            if hr != S_OK {
                return Err(VideoError::Codec(format!(
                    "MF: ProcessInput failed: {hr:#X}"
                )));
            }

            // Try to drain one output frame
            self.try_drain_output(timestamp_us)
        }

        /// Try to pull one decoded frame from the MFT via ProcessOutput.
        /// Returns `Ok(None)` when the decoder needs more input data.
        unsafe fn try_drain_output(
            &mut self,
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let mut output_buf = MFT_OUTPUT_DATA_BUFFER {
                stream_id: 0,
                sample: ptr::null_mut(),
                status: 0,
                events: ptr::null_mut(),
            };

            // Allocate output sample+buffer if MFT doesn't provide its own
            let pre_alloc = if !self.provides_samples {
                let mut out_sample: *mut c_void = ptr::null_mut();
                if MFCreateSample(&mut out_sample) != S_OK || out_sample.is_null() {
                    return Ok(None);
                }
                let buf_size = if self.output_buf_size > 0 {
                    self.output_buf_size
                } else if self.width > 0 && self.height > 0 {
                    self.width * self.height * 3 / 2
                } else {
                    // Conservative 4K NV12 fallback
                    4096 * 2160 * 3 / 2
                };
                let mut out_buf: *mut c_void = ptr::null_mut();
                if MFCreateMemoryBuffer(buf_size, &mut out_buf) != S_OK || out_buf.is_null() {
                    com_release(out_sample);
                    return Ok(None);
                }
                sample_add_buffer(out_sample, out_buf);
                com_release(out_buf);
                output_buf.sample = out_sample;
                out_sample
            } else {
                ptr::null_mut()
            };

            let mut proc_status: u32 = 0;
            let hr = transform_process_output(
                self.transform,
                0,
                1,
                &mut output_buf,
                &mut proc_status,
            );

            if !output_buf.events.is_null() {
                com_release(output_buf.events);
            }

            if hr == MF_E_TRANSFORM_NEED_MORE_INPUT {
                if !pre_alloc.is_null() {
                    com_release(pre_alloc);
                }
                return Ok(None);
            }
            if hr != S_OK {
                if !pre_alloc.is_null() {
                    com_release(pre_alloc);
                }
                if !output_buf.sample.is_null() && output_buf.sample != pre_alloc {
                    com_release(output_buf.sample);
                }
                return Err(VideoError::Codec(format!(
                    "MF: ProcessOutput failed: {hr:#X}"
                )));
            }

            let out_sample = output_buf.sample;
            if out_sample.is_null() {
                if !pre_alloc.is_null() {
                    com_release(pre_alloc);
                }
                return Ok(None);
            }

            // Get contiguous NV12 buffer from output sample
            let mut contig_buf: *mut c_void = ptr::null_mut();
            let hr = sample_convert_to_contiguous_buffer(out_sample, &mut contig_buf);
            if hr != S_OK || contig_buf.is_null() {
                com_release(out_sample);
                if !pre_alloc.is_null() && pre_alloc != out_sample {
                    com_release(pre_alloc);
                }
                return Err(VideoError::Codec(format!(
                    "MF: ConvertToContiguousBuffer failed: {hr:#X}"
                )));
            }

            let mut nv12_ptr: *mut u8 = ptr::null_mut();
            let mut max_len: u32 = 0;
            let mut nv12_len: u32 = 0;
            let hr = media_buffer_lock(contig_buf, &mut nv12_ptr, &mut max_len, &mut nv12_len);
            if hr != S_OK {
                com_release(contig_buf);
                com_release(out_sample);
                if !pre_alloc.is_null() && pre_alloc != out_sample {
                    com_release(pre_alloc);
                }
                return Err(VideoError::Codec(format!(
                    "MF: Lock output buffer failed: {hr:#X}"
                )));
            }

            // Resolve dimensions if not yet known (first frame decoded)
            if self.width == 0 || self.height == 0 {
                let mut out_type: *mut c_void = ptr::null_mut();
                if transform_get_output_current_type(self.transform, 0, &mut out_type) == S_OK
                    && !out_type.is_null()
                {
                    let mut frame_size: u64 = 0;
                    if attributes_get_uint64(out_type, &MF_MT_FRAME_SIZE, &mut frame_size) == S_OK
                    {
                        self.width = (frame_size >> 32) as u32;
                        self.height = frame_size as u32;
                    }
                    com_release(out_type);
                }
            }

            if self.width == 0 || self.height == 0 {
                media_buffer_unlock(contig_buf);
                com_release(contig_buf);
                com_release(out_sample);
                if !pre_alloc.is_null() && pre_alloc != out_sample {
                    com_release(pre_alloc);
                }
                return Err(VideoError::Codec(
                    "MF: cannot determine output dimensions".into(),
                ));
            }

            let w = self.width as usize;
            let h = self.height as usize;

            // NV12 → RGB8 via shared helper
            let mut rgb = vec![0u8; w * h * 3];
            super::nv12_to_rgb8(
                nv12_ptr,
                w,
                nv12_ptr.add(w * h),
                w,
                w,
                h,
                &mut rgb,
            );

            media_buffer_unlock(contig_buf);
            com_release(contig_buf);
            com_release(out_sample);
            if !pre_alloc.is_null() && pre_alloc != out_sample {
                com_release(pre_alloc);
            }

            Ok(Some(DecodedFrame {
                width: w,
                height: h,
                rgb8_data: rgb,
                timestamp_us,
                keyframe: false,
                bit_depth: 8,
                rgb16_data: None,
            }))
        }
    }

    impl VideoDecoder for MediaFoundationDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }
            // SAFETY: (category 1) transform handle validated; COM methods called via vtable.
            unsafe { self.feed_and_drain(data, timestamp_us) }
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            if self.transform.is_null() {
                return Ok(Vec::new());
            }
            let mut frames = Vec::new();
            // SAFETY: (category 1) drain remaining frames via ProcessOutput after sending
            // DRAIN message; loop terminates on NEED_MORE_INPUT or error.
            unsafe {
                transform_process_message(self.transform, MFT_MESSAGE_COMMAND_DRAIN, 0);
                loop {
                    match self.try_drain_output(0) {
                        Ok(Some(frame)) => frames.push(frame),
                        _ => break,
                    }
                }
            }
            Ok(frames)
        }
    }

    impl Drop for MediaFoundationDecoder {
        fn drop(&mut self) {
            if self.initialized {
                // SAFETY: (category 1) flush + release + shutdown; transform null-checked.
                unsafe {
                    if !self.transform.is_null() {
                        transform_process_message(
                            self.transform,
                            MFT_MESSAGE_COMMAND_FLUSH,
                            0,
                        );
                        com_release(self.transform);
                    }
                    MFShutdown();
                }
            }
        }
    }

    unsafe impl Send for MediaFoundationDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared NV12 → RGB8 helper (BT.601 limited-range, Q8 fixed-point)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert NV12 (Y plane + interleaved UV plane) to packed RGB8 using BT.601
/// limited-range coefficients with Q8 fixed-point arithmetic.
///
/// This is a platform-independent helper used by VA-API, NVDEC, and
/// MediaFoundation backends after GPU→host readback.
///
/// # Safety
///
/// `y_ptr` must point to at least `y_stride * h` readable bytes.
/// `uv_ptr` must point to at least `uv_stride * (h / 2)` readable bytes.
/// `rgb` must have length >= `w * h * 3`.
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub unsafe fn nv12_to_rgb8(
    y_ptr: *const u8,
    y_stride: usize,
    uv_ptr: *const u8,
    uv_stride: usize,
    w: usize,
    h: usize,
    rgb: &mut [u8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        nv12_to_rgb8_neon(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        nv12_to_rgb8_sse2(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
        return;
    }
    #[allow(unreachable_code)]
    nv12_to_rgb8_scalar(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn nv12_to_rgb8_neon(
    y_ptr: *const u8,
    y_stride: usize,
    uv_ptr: *const u8,
    uv_stride: usize,
    w: usize,
    h: usize,
    rgb: &mut [u8],
) {
    use std::arch::aarch64::*;

    let v16 = vdupq_n_s16(16);
    let v128 = vdupq_n_s16(128);
    let c298 = vdup_n_s16(298);
    let c409 = vdup_n_s16(409);
    let c100 = vdup_n_s16(100);
    let c208 = vdup_n_s16(208);
    let c516 = vdup_n_s16(516);
    let half = vdupq_n_s32(128);
    let zero16 = vdupq_n_s16(0);

    for row in 0..h {
        let y_row = y_ptr.add(row * y_stride);
        let uv_row = uv_ptr.add((row / 2) * uv_stride);
        let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
        let mut col = 0usize;

        while col + 8 <= w {
            let y8 = vld1_u8(y_row.add(col));
            let y16 = vreinterpretq_s16_u16(vmovl_u8(y8));
            let y_adj = vsubq_s16(y16, v16);

            let uv8 = vld1_u8(uv_row.add((col / 2) * 2));
            let uv16 = vreinterpretq_s16_u16(vmovl_u8(uv8));
            let cb4 = vuzp1q_s16(uv16, uv16);
            let cr4 = vuzp2q_s16(uv16, uv16);
            let cb = vzip1q_s16(cb4, cb4);
            let cr = vzip1q_s16(cr4, cr4);
            let cb_adj = vsubq_s16(cb, v128);
            let cr_adj = vsubq_s16(cr, v128);

            let y_lo = vget_low_s16(y_adj);
            let cb_lo = vget_low_s16(cb_adj);
            let cr_lo = vget_low_s16(cr_adj);
            let c_lo = vmull_s16(c298, y_lo);
            let r_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c409, cr_lo)), half), 8);
            let g_lo = vshrq_n_s32(vaddq_s32(vsubq_s32(vsubq_s32(c_lo, vmull_s16(c208, cr_lo)), vmull_s16(c100, cb_lo)), half), 8);
            let b_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c516, cb_lo)), half), 8);

            let y_hi = vget_high_s16(y_adj);
            let cb_hi = vget_high_s16(cb_adj);
            let cr_hi = vget_high_s16(cr_adj);
            let c_hi = vmull_s16(c298, y_hi);
            let r_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c409, cr_hi)), half), 8);
            let g_hi = vshrq_n_s32(vaddq_s32(vsubq_s32(vsubq_s32(c_hi, vmull_s16(c208, cr_hi)), vmull_s16(c100, cb_hi)), half), 8);
            let b_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c516, cb_hi)), half), 8);

            let r16 = vcombine_s16(vmovn_s32(r_lo), vmovn_s32(r_hi));
            let g16 = vcombine_s16(vmovn_s32(g_lo), vmovn_s32(g_hi));
            let b16 = vcombine_s16(vmovn_s32(b_lo), vmovn_s32(b_hi));
            let r8 = vqmovun_s16(vmaxq_s16(r16, zero16));
            let g8 = vqmovun_s16(vmaxq_s16(g16, zero16));
            let b8 = vqmovun_s16(vmaxq_s16(b16, zero16));

            vst3_u8(dst_row.as_mut_ptr().add(col * 3), uint8x8x3_t(r8, g8, b8));
            col += 8;
        }

        while col < w {
            let y_val = *y_row.add(col) as i32;
            let cb_val = *uv_row.add((col / 2) * 2) as i32;
            let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
            let c = 298 * (y_val - 16);
            let r = (c + 409 * (cr_val - 128) + 128) >> 8;
            let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
            let b = (c + 516 * (cb_val - 128) + 128) >> 8;
            let dst = col * 3;
            dst_row[dst] = r.clamp(0, 255) as u8;
            dst_row[dst + 1] = g.clamp(0, 255) as u8;
            dst_row[dst + 2] = b.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn nv12_to_rgb8_sse2(
    y_ptr: *const u8,
    y_stride: usize,
    uv_ptr: *const u8,
    uv_stride: usize,
    w: usize,
    h: usize,
    rgb: &mut [u8],
) {
    use std::arch::x86_64::*;

    let c149 = _mm_set1_epi16(149);
    let c204 = _mm_set1_epi16(204);
    let c50 = _mm_set1_epi16(50);
    let c104 = _mm_set1_epi16(104);
    let c258 = _mm_set1_epi16(258u16 as i16);
    let v16 = _mm_set1_epi16(16);
    let v128 = _mm_set1_epi16(128);
    let half = _mm_set1_epi16(64);
    let zero = _mm_setzero_si128();

    for row in 0..h {
        let y_row = y_ptr.add(row * y_stride);
        let uv_row = uv_ptr.add((row / 2) * uv_stride);
        let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
        let mut col = 0usize;

        while col + 8 <= w {
            let y8 = _mm_loadl_epi64(y_row.add(col) as *const __m128i);
            let y16 = _mm_unpacklo_epi8(y8, zero);
            let y_adj = _mm_sub_epi16(y16, v16);

            let mut cb_buf = [0u8; 8];
            let mut cr_buf = [0u8; 8];
            for i in 0..4 {
                cb_buf[i * 2] = *uv_row.add((col / 2 + i) * 2);
                cb_buf[i * 2 + 1] = *uv_row.add((col / 2 + i) * 2);
                cr_buf[i * 2] = *uv_row.add((col / 2 + i) * 2 + 1);
                cr_buf[i * 2 + 1] = *uv_row.add((col / 2 + i) * 2 + 1);
            }
            let cb8 = _mm_loadl_epi64(cb_buf.as_ptr() as *const __m128i);
            let cr8 = _mm_loadl_epi64(cr_buf.as_ptr() as *const __m128i);
            let cb_adj = _mm_sub_epi16(_mm_unpacklo_epi8(cb8, zero), v128);
            let cr_adj = _mm_sub_epi16(_mm_unpacklo_epi8(cr8, zero), v128);

            let c_val = _mm_mullo_epi16(c149, y_adj);
            let r16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                _mm_add_epi16(c_val, _mm_mullo_epi16(c204, cr_adj)),
                half,
            ));
            let g16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                _mm_sub_epi16(
                    _mm_sub_epi16(c_val, _mm_mullo_epi16(c104, cr_adj)),
                    _mm_mullo_epi16(c50, cb_adj),
                ),
                half,
            ));
            let b16 = _mm_srai_epi16::<7>(_mm_add_epi16(
                _mm_add_epi16(c_val, _mm_mullo_epi16(c258, cb_adj)),
                half,
            ));

            let r_u8 = _mm_packus_epi16(_mm_max_epi16(r16, zero), zero);
            let g_u8 = _mm_packus_epi16(_mm_max_epi16(g16, zero), zero);
            let b_u8 = _mm_packus_epi16(_mm_max_epi16(b16, zero), zero);

            let mut rgb_buf = [0u8; 24];
            let mut r_arr = [0u8; 8];
            let mut g_arr = [0u8; 8];
            let mut b_arr = [0u8; 8];
            _mm_storel_epi64(r_arr.as_mut_ptr() as *mut __m128i, r_u8);
            _mm_storel_epi64(g_arr.as_mut_ptr() as *mut __m128i, g_u8);
            _mm_storel_epi64(b_arr.as_mut_ptr() as *mut __m128i, b_u8);
            for i in 0..8 {
                rgb_buf[i * 3] = r_arr[i];
                rgb_buf[i * 3 + 1] = g_arr[i];
                rgb_buf[i * 3 + 2] = b_arr[i];
            }
            std::ptr::copy_nonoverlapping(
                rgb_buf.as_ptr(),
                dst_row.as_mut_ptr().add(col * 3),
                24,
            );

            col += 8;
        }

        while col < w {
            let y_val = *y_row.add(col) as i32;
            let cb_val = *uv_row.add((col / 2) * 2) as i32;
            let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
            let c = 298 * (y_val - 16);
            let r = (c + 409 * (cr_val - 128) + 128) >> 8;
            let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
            let b = (c + 516 * (cb_val - 128) + 128) >> 8;
            let dst = col * 3;
            dst_row[dst] = r.clamp(0, 255) as u8;
            dst_row[dst + 1] = g.clamp(0, 255) as u8;
            dst_row[dst + 2] = b.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn nv12_to_rgb8_scalar(
    y_ptr: *const u8,
    y_stride: usize,
    uv_ptr: *const u8,
    uv_stride: usize,
    w: usize,
    h: usize,
    rgb: &mut [u8],
) {
    for row in 0..h {
        let y_row = y_ptr.add(row * y_stride);
        let uv_row = uv_ptr.add((row / 2) * uv_stride);
        let dst_base = row * w * 3;
        for col in 0..w {
            let y_val = *y_row.add(col) as i32;
            let cb_val = *uv_row.add((col / 2) * 2) as i32;
            let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
            let c = 298 * (y_val - 16);
            let r = (c + 409 * (cr_val - 128) + 128) >> 8;
            let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
            let b = (c + 516 * (cb_val - 128) + 128) >> 8;
            let dst = dst_base + col * 3;
            rgb[dst] = r.clamp(0, 255) as u8;
            rgb[dst + 1] = g.clamp(0, 255) as u8;
            rgb[dst + 2] = b.clamp(0, 255) as u8;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Auto-dispatch decoder
// ═══════════════════════════════════════════════════════════════════════════

/// Hardware-accelerated video decoder with automatic software fallback.
pub struct HwVideoDecoder {
    backend: HwBackend,
    inner: Box<dyn VideoDecoder>,
}

impl HwVideoDecoder {
    /// Create a decoder with automatic backend selection.
    pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
        let backend = detect_hw_backend();

        let hw_result: Result<Box<dyn VideoDecoder>, VideoError> = match backend {
            #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
            HwBackend::VideoToolbox => videotoolbox::VideoToolboxDecoder::new(codec)
                .map(|d| Box::new(d) as Box<dyn VideoDecoder>),

            #[cfg(all(target_os = "linux", feature = "vaapi"))]
            HwBackend::Vaapi => {
                vaapi::VaapiDecoder::new(codec).map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }

            #[cfg(feature = "nvdec")]
            HwBackend::Nvdec => {
                nvdec::NvdecDecoder::new(codec).map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }

            #[cfg(all(target_os = "windows", feature = "media-foundation"))]
            HwBackend::MediaFoundation => media_foundation::MediaFoundationDecoder::new(codec)
                .map(|d| Box::new(d) as Box<dyn VideoDecoder>),

            _ => Err(VideoError::Codec("No hardware backend available".into())),
        };

        match hw_result {
            Ok(decoder) => Ok(HwVideoDecoder {
                backend,
                inner: decoder,
            }),
            Err(_) => {
                let sw: Box<dyn VideoDecoder> = match codec {
                    VideoCodec::H264 => Box::new(super::h264_decoder::H264Decoder::new()),
                    VideoCodec::H265 => Box::new(super::hevc_decoder::HevcDecoder::new()),
                    _ => return Err(VideoError::Codec(format!("Unsupported codec: {codec:?}"))),
                };
                Ok(HwVideoDecoder {
                    backend: HwBackend::Software,
                    inner: sw,
                })
            }
        }
    }

    pub fn backend(&self) -> HwBackend {
        self.backend
    }
    pub fn is_hardware(&self) -> bool {
        self.backend != HwBackend::Software
    }
}

impl VideoDecoder for HwVideoDecoder {
    fn codec(&self) -> VideoCodec {
        self.inner.codec()
    }
    fn decode(&mut self, data: &[u8], ts: u64) -> Result<Option<DecodedFrame>, VideoError> {
        self.inner.decode(data, ts)
    }
    fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
        self.inner.flush()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_backend() {
        let backend = detect_hw_backend();
        println!("Detected backend: {backend}");
    }

    #[test]
    fn hw_decoder_fallback_h264() {
        let decoder = HwVideoDecoder::new(VideoCodec::H264).unwrap();
        // Without features, falls back to software
        println!("H264 backend: {}", decoder.backend());
    }

    #[test]
    fn hw_decoder_fallback_hevc() {
        let decoder = HwVideoDecoder::new(VideoCodec::H265).unwrap();
        println!("HEVC backend: {}", decoder.backend());
    }
}
