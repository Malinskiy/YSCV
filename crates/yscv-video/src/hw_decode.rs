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
                let g_lo = vshrq_n_s32(
                    vaddq_s32(
                        vsubq_s32(
                            vsubq_s32(c_lo, vmull_s16(c208, cr_lo)),
                            vmull_s16(c100, cb_lo),
                        ),
                        half,
                    ),
                    8,
                );
                let b_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c516, cb_lo)), half), 8);

                // High 4 pixels
                let y_hi = vget_high_s16(y_adj);
                let cb_hi = vget_high_s16(cb_adj);
                let cr_hi = vget_high_s16(cr_adj);
                let c_hi = vmull_s16(c298, y_hi);
                let r_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c409, cr_hi)), half), 8);
                let g_hi = vshrq_n_s32(
                    vaddq_s32(
                        vsubq_s32(
                            vsubq_s32(c_hi, vmull_s16(c208, cr_hi)),
                            vmull_s16(c100, cb_hi),
                        ),
                        half,
                    ),
                    8,
                );
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
#[allow(
    unsafe_code,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn,
    non_snake_case,
    non_upper_case_globals
)]
pub mod vaapi {
    use super::*;
    use crate::h264_bitstream::BitstreamReader;
    use crate::h264_params::{Pps, SliceHeader, Sps};
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

    const VA_PROFILE_H264_CONSTRAINED_BASELINE: VAProfile = 13;
    const VA_PROFILE_H264_MAIN: VAProfile = 6;
    const VA_PROFILE_H264_HIGH: VAProfile = 7;
    const VA_PROFILE_HEVC_MAIN: VAProfile = 12;
    const VA_ENTRYPOINT_VLD: VAEntrypoint = 1;
    const VA_STATUS_SUCCESS: VAStatus = 0;
    const VA_PADDING_LOW: usize = 4;
    const VA_PADDING_MEDIUM: usize = 8;

    const VAPictureParameterBufferType: i32 = 0;
    const VAIQMatrixBufferType: i32 = 1;
    const VASliceParameterBufferType: i32 = 4;
    // VASliceDataBufferType = 5 already exists

    const VA_PICTURE_H264_INVALID: u32 = 0x0000_0001;
    const VA_PICTURE_H264_TOP_FIELD: u32 = 0x0000_0002;
    const VA_PICTURE_H264_BOTTOM_FIELD: u32 = 0x0000_0004;
    const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
    const VA_PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x0000_0010;

    const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

    const VA_INVALID_ID: u32 = 0xFFFF_FFFF;

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
        va_reserved: [u32; 4],
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
        va_reserved: [u32; 4],
    }

    const VA_RT_FORMAT_YUV420: u32 = 0x00000001;
    const VASliceDataBufferType: i32 = 5;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct VAPictureH264 {
        picture_id: VASurfaceID,
        frame_idx: u32,
        flags: u32,
        TopFieldOrderCnt: i32,
        BottomFieldOrderCnt: i32,
        va_reserved: [u32; VA_PADDING_LOW],
    }

    impl Default for VAPictureH264 {
        fn default() -> Self {
            Self {
                picture_id: VA_INVALID_ID,
                frame_idx: 0,
                flags: VA_PICTURE_H264_INVALID,
                TopFieldOrderCnt: 0,
                BottomFieldOrderCnt: 0,
                va_reserved: [0; VA_PADDING_LOW],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct VAPictureParameterBufferH264 {
        CurrPic: VAPictureH264,
        ReferenceFrames: [VAPictureH264; 16],
        picture_width_in_mbs_minus1: u16,
        picture_height_in_mbs_minus1: u16,
        bit_depth_luma_minus8: u8,
        bit_depth_chroma_minus8: u8,
        num_ref_frames: u8,
        seq_fields: u32,
        num_slice_groups_minus1: u8,
        slice_group_map_type: u8,
        slice_group_change_rate_minus1: u16,
        pic_init_qp_minus26: i8,
        pic_init_qs_minus26: i8,
        chroma_qp_index_offset: i8,
        second_chroma_qp_index_offset: i8,
        pic_fields: u32,
        frame_num: u16,
        va_reserved: [u32; VA_PADDING_MEDIUM],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct VAIQMatrixBufferH264 {
        ScalingList4x4: [[u8; 16]; 6],
        ScalingList8x8: [[u8; 64]; 2],
        va_reserved: [u32; VA_PADDING_LOW],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct VASliceParameterBufferH264 {
        slice_data_size: u32,
        slice_data_offset: u32,
        slice_data_flag: u32,
        slice_data_bit_offset: u16,
        first_mb_in_slice: u16,
        slice_type: u8,
        direct_spatial_mv_pred_flag: u8,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
        cabac_init_idc: u8,
        slice_qp_delta: i8,
        disable_deblocking_filter_idc: u8,
        slice_alpha_c0_offset_div2: i8,
        slice_beta_offset_div2: i8,
        RefPicList0: [VAPictureH264; 32],
        RefPicList1: [VAPictureH264; 32],
        luma_log2_weight_denom: u8,
        chroma_log2_weight_denom: u8,
        luma_weight_l0_flag: u8,
        luma_weight_l0: [i16; 32],
        luma_offset_l0: [i16; 32],
        chroma_weight_l0_flag: u8,
        chroma_weight_l0: [[i16; 2]; 32],
        chroma_offset_l0: [[i16; 2]; 32],
        luma_weight_l1_flag: u8,
        luma_weight_l1: [i16; 32],
        luma_offset_l1: [i16; 32],
        chroma_weight_l1_flag: u8,
        chroma_weight_l1: [[i16; 2]; 32],
        chroma_offset_l1: [[i16; 2]; 32],
        va_reserved: [u32; VA_PADDING_LOW],
    }

    #[inline]
    fn check_va(status: VAStatus, op: &str) -> Result<(), VideoError> {
        if status == VA_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(VideoError::Codec(format!("VA-API: {op} failed: {status}")))
        }
    }

    fn pack_seq_fields(sps: &Sps) -> u32 {
        // Bitfield layout matches VAPictureParameterBufferH264.seq_fields.bits
        (sps.chroma_format_idc & 0x3)          // bits 0-1: chroma_format_idc
            // bit 2: separate_colour_plane_flag = 0 (not parsed)
            // bit 3: gaps_in_frame_num_value_allowed_flag = 0 (not parsed)
            | ((sps.frame_mbs_only_flag as u32) << 4)
            | ((sps.mb_adaptive_frame_field_flag as u32) << 5)
            | (1u32 << 6)                       // direct_8x8_inference_flag = 1
            | (((sps.level_idc >= 31) as u32) << 7) // MinLumaBiPredSize8x8
            | (((sps.log2_max_frame_num.saturating_sub(4)) & 0xF) << 8)
            | ((sps.pic_order_cnt_type & 0x3) << 12)
            | (((sps.log2_max_pic_order_cnt_lsb.saturating_sub(4)) & 0xF) << 14)
            // bit 18: delta_pic_order_always_zero_flag = 0 (not parsed)
    }

    fn pack_pic_fields(pps: &Pps, sh: &SliceHeader, nal_ref_idc: u8) -> u32 {
        // Bitfield layout matches VAPictureParameterBufferH264.pic_fields.bits
        (pps.entropy_coding_mode_flag as u32)
            | ((pps.weighted_pred_flag as u32) << 1)
            | ((pps.weighted_bipred_idc & 0x3) << 2)
            | ((pps.transform_8x8_mode_flag as u32) << 4)
            | ((sh.field_pic_flag as u32) << 5)
            // bit 6: constrained_intra_pred_flag = 0 (not parsed)
            // bit 7: pic_order_present_flag = 0 (not parsed)
            | ((pps.deblocking_filter_control_present_flag as u32) << 8)
            // bit 9: redundant_pic_cnt_present_flag = 0
            | (((nal_ref_idc != 0) as u32) << 10)
    }

    /// Dynamically-loaded libva function pointers.
    struct VaLib {
        _lib: libloading::Library,
        va_initialize: unsafe extern "C" fn(VADisplay, *mut i32, *mut i32) -> VAStatus,
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
        va_begin_picture: unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus,
        va_create_buffer: unsafe extern "C" fn(
            VADisplay,
            VAContextID,
            i32,
            u32,
            u32,
            *const c_void,
            *mut VABufferID,
        ) -> VAStatus,
        va_render_picture:
            unsafe extern "C" fn(VADisplay, VAContextID, *mut VABufferID, i32) -> VAStatus,
        va_end_picture: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
        va_sync_surface: unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus,
        va_derive_image: unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VAImage) -> VAStatus,
        va_map_buffer: unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus,
        va_unmap_buffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
        va_destroy_image: unsafe extern "C" fn(VADisplay, u32) -> VAStatus,
        va_destroy_buffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
        va_destroy_surfaces: unsafe extern "C" fn(VADisplay, *mut VASurfaceID, i32) -> VAStatus,
        va_destroy_config: unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus,
        va_destroy_context: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
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
                    .get::<unsafe extern "C" fn(
                        VADisplay,
                        VAProfile,
                        VAEntrypoint,
                        *const c_void,
                        i32,
                        *mut VAConfigID,
                    ) -> VAStatus>(b"vaCreateConfig\0")
                    .ok()?;
                let va_create_surfaces = *lib
                    .get::<unsafe extern "C" fn(
                        VADisplay,
                        u32,
                        u32,
                        u32,
                        *mut VASurfaceID,
                        u32,
                        *const c_void,
                        u32,
                    ) -> VAStatus>(b"vaCreateSurfaces\0")
                    .ok()?;
                let va_create_context = *lib
                    .get::<unsafe extern "C" fn(
                        VADisplay,
                        VAConfigID,
                        i32,
                        i32,
                        i32,
                        *mut VASurfaceID,
                        i32,
                        *mut VAContextID,
                    ) -> VAStatus>(b"vaCreateContext\0")
                    .ok()?;
                let va_begin_picture = *lib
                    .get::<unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus>(
                        b"vaBeginPicture\0",
                    )
                    .ok()?;
                let va_create_buffer = *lib
                    .get::<unsafe extern "C" fn(
                        VADisplay,
                        VAContextID,
                        i32,
                        u32,
                        u32,
                        *const c_void,
                        *mut VABufferID,
                    ) -> VAStatus>(b"vaCreateBuffer\0")
                    .ok()?;
                let va_render_picture =
                    *lib.get::<unsafe extern "C" fn(
                        VADisplay,
                        VAContextID,
                        *mut VABufferID,
                        i32,
                    ) -> VAStatus>(b"vaRenderPicture\0")
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
                    .get::<unsafe extern "C" fn(VADisplay, u32) -> VAStatus>(b"vaDestroyImage\0")
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
            let lib = unsafe { libloading::Library::new("libva-drm.so.2") }.ok()?;
            // SAFETY: symbol signature matches the libva-drm C ABI.
            unsafe {
                let va_get_display_drm = *lib
                    .get::<unsafe extern "C" fn(i32) -> VADisplay>(b"vaGetDisplayDRM\0")
                    .ok()?;
                Some(Self {
                    _lib: lib,
                    va_get_display_drm,
                })
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct DpbEntry {
        surface_idx: usize,
        frame_num: u32,
        top_poc: i32,
        bottom_poc: i32,
        is_long_term: bool,
        is_reference: bool,
    }

    #[derive(Debug)]
    struct Dpb {
        entries: Vec<DpbEntry>,
        max_size: usize,
        poc_counter: i32,
        prev_poc_lsb: u32,
        prev_poc_msb: i32,
    }

    impl Dpb {
        fn new(max_size: usize) -> Self {
            Self {
                entries: Vec::new(),
                max_size,
                poc_counter: 0,
                prev_poc_lsb: 0,
                prev_poc_msb: 0,
            }
        }

        fn clear(&mut self) -> Vec<usize> {
            let released = self.entries.iter().map(|e| e.surface_idx).collect();
            self.entries.clear();
            self.poc_counter = 0;
            self.prev_poc_lsb = 0;
            self.prev_poc_msb = 0;
            released
        }

        fn add_reference(
            &mut self,
            surface_idx: usize,
            frame_num: u32,
            top_poc: i32,
            bottom_poc: i32,
        ) {
            self.entries.push(DpbEntry {
                surface_idx,
                frame_num,
                top_poc,
                bottom_poc,
                is_long_term: false,
                is_reference: true,
            });
        }

        fn sliding_window_evict(&mut self, surface_in_use: &mut [bool]) {
            if self.entries.len() < self.max_size {
                return;
            }
            if let Some((idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .find(|(_, entry)| entry.is_reference && !entry.is_long_term)
            {
                let removed = self.entries.remove(idx);
                if removed.surface_idx < surface_in_use.len() {
                    surface_in_use[removed.surface_idx] = false;
                }
                return;
            }
            if let Some(removed) = self.entries.first().copied() {
                self.entries.remove(0);
                if removed.surface_idx < surface_in_use.len() {
                    surface_in_use[removed.surface_idx] = false;
                }
            }
        }

        fn to_va_picture(entry: &DpbEntry, surfaces: &[VASurfaceID]) -> VAPictureH264 {
            let mut flags = if entry.is_long_term {
                VA_PICTURE_H264_LONG_TERM_REFERENCE
            } else {
                VA_PICTURE_H264_SHORT_TERM_REFERENCE
            };
            if entry.top_poc != entry.bottom_poc {
                flags |= VA_PICTURE_H264_TOP_FIELD | VA_PICTURE_H264_BOTTOM_FIELD;
            }
            VAPictureH264 {
                picture_id: surfaces[entry.surface_idx],
                frame_idx: entry.frame_num,
                flags,
                TopFieldOrderCnt: entry.top_poc,
                BottomFieldOrderCnt: entry.bottom_poc,
                va_reserved: [0; VA_PADDING_LOW],
            }
        }

        fn build_va_reference_frames(&self, surfaces: &[VASurfaceID]) -> [VAPictureH264; 16] {
            let mut out = [VAPictureH264::default(); 16];
            for (i, entry) in self
                .entries
                .iter()
                .filter(|e| e.is_reference)
                .take(16)
                .enumerate()
            {
                out[i] = Self::to_va_picture(entry, surfaces);
            }
            out
        }

        fn build_ref_pic_list0(
            &self,
            surfaces: &[VASurfaceID],
            slice_type: u32,
            current_poc: i32,
        ) -> [VAPictureH264; 32] {
            let mut out = [VAPictureH264::default(); 32];
            let normalized = slice_type % 5;
            let mut refs: Vec<DpbEntry> = self
                .entries
                .iter()
                .copied()
                .filter(|e| e.is_reference)
                .collect();

            if normalized == 1 {
                let mut le: Vec<DpbEntry> = refs
                    .iter()
                    .copied()
                    .filter(|e| !e.is_long_term && e.top_poc <= current_poc)
                    .collect();
                let mut gt: Vec<DpbEntry> = refs
                    .iter()
                    .copied()
                    .filter(|e| !e.is_long_term && e.top_poc > current_poc)
                    .collect();
                let mut lt: Vec<DpbEntry> =
                    refs.iter().copied().filter(|e| e.is_long_term).collect();
                le.sort_by(|a, b| b.top_poc.cmp(&a.top_poc));
                gt.sort_by(|a, b| a.top_poc.cmp(&b.top_poc));
                lt.sort_by(|a, b| a.frame_num.cmp(&b.frame_num));
                refs.clear();
                refs.extend(le);
                refs.extend(gt);
                refs.extend(lt);
            } else {
                let mut st: Vec<DpbEntry> =
                    refs.iter().copied().filter(|e| !e.is_long_term).collect();
                let mut lt: Vec<DpbEntry> =
                    refs.iter().copied().filter(|e| e.is_long_term).collect();
                st.sort_by(|a, b| b.frame_num.cmp(&a.frame_num));
                lt.sort_by(|a, b| a.frame_num.cmp(&b.frame_num));
                refs.clear();
                refs.extend(st);
                refs.extend(lt);
            }

            for (i, entry) in refs.iter().take(32).enumerate() {
                out[i] = Self::to_va_picture(entry, surfaces);
            }
            out
        }

        fn build_ref_pic_list1(
            &self,
            surfaces: &[VASurfaceID],
            slice_type: u32,
            current_poc: i32,
        ) -> [VAPictureH264; 32] {
            let normalized = slice_type % 5;
            if normalized != 1 {
                return [VAPictureH264::default(); 32];
            }
            let mut list = self.build_ref_pic_list0(surfaces, slice_type, current_poc);
            list.reverse();
            list
        }

        fn compute_poc(&mut self, sps: &Sps, sh: &SliceHeader, is_idr: bool) -> (i32, i32) {
            if sps.pic_order_cnt_type == 0 {
                let max_poc_lsb = 1u32 << sps.log2_max_pic_order_cnt_lsb.min(30);
                let poc_msb = if is_idr {
                    0
                } else if sh.pic_order_cnt_lsb < self.prev_poc_lsb
                    && (self.prev_poc_lsb - sh.pic_order_cnt_lsb) >= (max_poc_lsb / 2)
                {
                    self.prev_poc_msb + max_poc_lsb as i32
                } else if sh.pic_order_cnt_lsb > self.prev_poc_lsb
                    && (sh.pic_order_cnt_lsb - self.prev_poc_lsb) > (max_poc_lsb / 2)
                {
                    self.prev_poc_msb - max_poc_lsb as i32
                } else {
                    self.prev_poc_msb
                };
                let top = poc_msb + sh.pic_order_cnt_lsb as i32;
                let bottom = if sh.field_pic_flag {
                    top
                } else {
                    top + sh.delta_pic_order_cnt_bottom
                };
                self.prev_poc_msb = poc_msb;
                self.prev_poc_lsb = sh.pic_order_cnt_lsb;
                (top, bottom)
            } else {
                if is_idr {
                    self.poc_counter = 0;
                }
                let top = self.poc_counter;
                self.poc_counter = self.poc_counter.saturating_add(2);
                (top, top)
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
        sps: Option<Sps>,
        pps: Option<Pps>,
        dpb: Dpb,
        surface_in_use: Vec<bool>,
        current_surface_idx: usize,
        frame_counter: u64,
        h264_profile: VAProfile,
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
                let status = (va.va_initialize)(display, &mut major, &mut minor);
                if status != VA_STATUS_SUCCESS {
                    return Self::with_sw_fallback(codec);
                }

                let profile = match codec {
                    VideoCodec::H264 => VA_PROFILE_H264_HIGH,
                    VideoCodec::H265 => VA_PROFILE_HEVC_MAIN,
                    _ => return Err(VideoError::Codec("Unsupported codec".into())),
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
                    sps: None,
                    pps: None,
                    dpb: Dpb::new(16),
                    surface_in_use: Vec::new(),
                    current_surface_idx: 0,
                    frame_counter: 0,
                    h264_profile: profile,
                })
            }
        }

        fn with_sw_fallback(codec: VideoCodec) -> Result<Self, VideoError> {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                VideoCodec::H265 => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
                _ => return Err(VideoError::Codec("Unsupported codec".into())),
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
                sps: None,
                pps: None,
                dpb: Dpb::new(16),
                surface_in_use: Vec::new(),
                current_surface_idx: 0,
                frame_counter: 0,
                h264_profile: VA_PROFILE_H264_HIGH,
            })
        }

        /// Create surfaces and context for the given resolution.
        unsafe fn create_surfaces(&mut self, width: u32, height: u32) -> Result<(), VideoError> {
            let va = self.va.as_ref().unwrap();
            self.width = width;
            self.height = height;
            let num_surfaces: u32 = if let Some(ref sps) = self.sps {
                sps.max_num_ref_frames.min(16).saturating_add(2)
            } else {
                4
            };
            self.surfaces = vec![0u32; num_surfaces as usize];
            self.surface_in_use = vec![false; num_surfaces as usize];
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
            self.current_surface_idx = 0;
            Ok(())
        }

        unsafe fn destroy_surfaces_and_context(&mut self) {
            let va = self.va.as_ref().unwrap();
            if self.context != 0 {
                (va.va_destroy_context)(self.display, self.context);
                self.context = 0;
            }
            if !self.surfaces.is_empty() {
                (va.va_destroy_surfaces)(
                    self.display,
                    self.surfaces.as_mut_ptr(),
                    self.surfaces.len() as i32,
                );
                self.surfaces.clear();
            }
            self.surface_in_use.clear();
            self.surfaces_created = false;
        }

        fn select_h264_profile(profile_idc: u8) -> VAProfile {
            match profile_idc {
                66 => VA_PROFILE_H264_CONSTRAINED_BASELINE,
                77 => VA_PROFILE_H264_MAIN,
                100 => VA_PROFILE_H264_HIGH,
                _ => VA_PROFILE_H264_HIGH,
            }
        }

        unsafe fn recreate_config(&mut self, profile: VAProfile) -> Result<(), VideoError> {
            let va = self.va.as_ref().unwrap();
            if self.config != 0 {
                (va.va_destroy_config)(self.display, self.config);
                self.config = 0;
            }
            let mut config_id: VAConfigID = 0;
            check_va(
                (va.va_create_config)(
                    self.display,
                    profile,
                    VA_ENTRYPOINT_VLD,
                    ptr::null(),
                    0,
                    &mut config_id,
                ),
                "vaCreateConfig",
            )?;
            self.config = config_id;
            self.h264_profile = profile;
            Ok(())
        }

        fn find_free_surface(&self) -> Option<usize> {
            self.surface_in_use.iter().position(|&in_use| !in_use)
        }

        unsafe fn readback_surface(
            &self,
            surface: VASurfaceID,
            timestamp_us: u64,
            keyframe: bool,
        ) -> Result<DecodedFrame, VideoError> {
            let va = self.va.as_ref().unwrap();
            check_va((va.va_sync_surface)(self.display, surface), "vaSyncSurface")?;

            let mut image: VAImage = std::mem::zeroed();
            check_va(
                (va.va_derive_image)(self.display, surface, &mut image),
                "vaDeriveImage",
            )?;

            let mut buf_ptr: *mut c_void = ptr::null_mut();
            if let Err(e) = check_va(
                (va.va_map_buffer)(self.display, image.buf, &mut buf_ptr),
                "vaMapBuffer",
            ) {
                (va.va_destroy_image)(self.display, image.image_id);
                return Err(e);
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

            Ok(DecodedFrame {
                width: w,
                height: h,
                rgb8_data: rgb,
                timestamp_us,
                keyframe,
                bit_depth: 8,
                rgb16_data: None,
            })
        }

        unsafe fn decode_legacy_slice(
            &mut self,
            slice_data: &[u8],
            surface_idx: usize,
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let va = self.va.as_ref().unwrap();
            let surface = self.surfaces[surface_idx % self.surfaces.len()];

            check_va(
                (va.va_begin_picture)(self.display, self.context, surface),
                "vaBeginPicture",
            )?;

            let mut slice_buf: VABufferID = 0;
            if let Err(e) = check_va(
                (va.va_create_buffer)(
                    self.display,
                    self.context,
                    VASliceDataBufferType,
                    slice_data.len() as u32,
                    1,
                    slice_data.as_ptr() as *const c_void,
                    &mut slice_buf,
                ),
                "vaCreateBuffer(SliceData)",
            ) {
                (va.va_end_picture)(self.display, self.context);
                return Err(e);
            }

            let mut slice_bufs = [slice_buf];
            if let Err(e) = check_va(
                (va.va_render_picture)(self.display, self.context, slice_bufs.as_mut_ptr(), 1),
                "vaRenderPicture",
            ) {
                (va.va_destroy_buffer)(self.display, slice_buf);
                (va.va_end_picture)(self.display, self.context);
                return Err(e);
            }

            if let Err(e) = check_va(
                (va.va_end_picture)(self.display, self.context),
                "vaEndPicture",
            ) {
                (va.va_destroy_buffer)(self.display, slice_buf);
                return Err(e);
            }
            let frame = self.readback_surface(surface, timestamp_us, false)?;
            (va.va_destroy_buffer)(self.display, slice_buf);
            Ok(Some(frame))
        }

        unsafe fn decode_picture(
            &mut self,
            slice_nals: &[&crate::NalUnit],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let sps = self
                .sps
                .as_ref()
                .ok_or_else(|| VideoError::Codec("no SPS".into()))?;
            let pps = self
                .pps
                .as_ref()
                .ok_or_else(|| VideoError::Codec("no PPS".into()))?;

            let mut surface_idx = self.find_free_surface();
            if surface_idx.is_none() {
                self.dpb.sliding_window_evict(&mut self.surface_in_use);
                surface_idx = self.find_free_surface();
            }
            let surf_idx =
                surface_idx.ok_or_else(|| VideoError::Codec("no free surface".into()))?;
            let surface = self.surfaces[surf_idx];
            self.surface_in_use[surf_idx] = true;

            let first_nal = slice_nals[0];
            let nal_ref_idc = (first_nal.data[0] >> 5) & 0x3;
            let nal_type = first_nal.data[0] & 0x1F;
            let is_idr = nal_type == 5;

            let first_rbsp = crate::h264_params::remove_emulation_prevention(&first_nal.data[1..]);
            let mut first_reader = BitstreamReader::new(&first_rbsp);
            let first_sh =
                crate::h264_params::parse_slice_header(&mut first_reader, sps, pps, is_idr)?;

            if is_idr {
                for idx in self.dpb.clear() {
                    if idx < self.surface_in_use.len() {
                        self.surface_in_use[idx] = false;
                    }
                }
                self.surface_in_use[surf_idx] = true;
            }

            let (top_poc, bottom_poc) = self.dpb.compute_poc(sps, &first_sh, is_idr);

            let mut pic_param: VAPictureParameterBufferH264 = std::mem::zeroed();
            pic_param.CurrPic = VAPictureH264 {
                picture_id: surface,
                frame_idx: first_sh.frame_num,
                flags: 0,
                TopFieldOrderCnt: top_poc,
                BottomFieldOrderCnt: bottom_poc,
                va_reserved: [0; VA_PADDING_LOW],
            };
            pic_param.ReferenceFrames = self.dpb.build_va_reference_frames(&self.surfaces);
            pic_param.picture_width_in_mbs_minus1 = (sps.pic_width_in_mbs.saturating_sub(1)) as u16;
            let pic_height_in_mbs = if sps.frame_mbs_only_flag {
                sps.pic_height_in_map_units
            } else {
                sps.pic_height_in_map_units.saturating_mul(2)
            };
            pic_param.picture_height_in_mbs_minus1 = pic_height_in_mbs.saturating_sub(1) as u16;
            pic_param.bit_depth_luma_minus8 = sps.bit_depth_luma.saturating_sub(8) as u8;
            pic_param.bit_depth_chroma_minus8 = sps.bit_depth_chroma.saturating_sub(8) as u8;
            pic_param.num_ref_frames = sps.max_num_ref_frames.min(16) as u8;
            pic_param.seq_fields = pack_seq_fields(sps);
            pic_param.pic_init_qp_minus26 = (pps.pic_init_qp - 26) as i8;
            pic_param.pic_fields = pack_pic_fields(pps, &first_sh, nal_ref_idc);
            pic_param.frame_num = first_sh.frame_num as u16;

            let mut iq_matrix: VAIQMatrixBufferH264 = std::mem::zeroed();
            for i in 0..6 {
                for j in 0..16 {
                    iq_matrix.ScalingList4x4[i][j] = sps.scaling_list_4x4[i][j] as u8;
                }
            }
            for j in 0..64 {
                iq_matrix.ScalingList8x8[0][j] = sps.scaling_list_8x8[0][j] as u8;
                iq_matrix.ScalingList8x8[1][j] = sps.scaling_list_8x8[3][j] as u8;
            }

            let va = self.va.as_ref().unwrap();
            let mut created_buffers: Vec<VABufferID> = Vec::new();
            let mut picture_started = false;

            let result = (|| -> Result<DecodedFrame, VideoError> {
                let mut pic_param_buf: VABufferID = 0;
                check_va(
                    (va.va_create_buffer)(
                        self.display,
                        self.context,
                        VAPictureParameterBufferType,
                        std::mem::size_of::<VAPictureParameterBufferH264>() as u32,
                        1,
                        &pic_param as *const _ as *const c_void,
                        &mut pic_param_buf,
                    ),
                    "vaCreateBuffer(PictureParameter)",
                )?;
                created_buffers.push(pic_param_buf);

                let mut iq_buf: VABufferID = 0;
                check_va(
                    (va.va_create_buffer)(
                        self.display,
                        self.context,
                        VAIQMatrixBufferType,
                        std::mem::size_of::<VAIQMatrixBufferH264>() as u32,
                        1,
                        &iq_matrix as *const _ as *const c_void,
                        &mut iq_buf,
                    ),
                    "vaCreateBuffer(IQMatrix)",
                )?;
                created_buffers.push(iq_buf);

                check_va(
                    (va.va_begin_picture)(self.display, self.context, surface),
                    "vaBeginPicture",
                )?;
                picture_started = true;

                let mut param_bufs = [pic_param_buf, iq_buf];
                check_va(
                    (va.va_render_picture)(self.display, self.context, param_bufs.as_mut_ptr(), 2),
                    "vaRenderPicture(params)",
                )?;

                for slice_nal in slice_nals {
                    let rbsp =
                        crate::h264_params::remove_emulation_prevention(&slice_nal.data[1..]);
                    let mut reader = BitstreamReader::new(&rbsp);
                    let sh = crate::h264_params::parse_slice_header(&mut reader, sps, pps, is_idr)?;

                    let mut sp: VASliceParameterBufferH264 = std::mem::zeroed();
                    sp.RefPicList0 =
                        self.dpb
                            .build_ref_pic_list0(&self.surfaces, sh.slice_type, top_poc);
                    sp.RefPicList1 =
                        self.dpb
                            .build_ref_pic_list1(&self.surfaces, sh.slice_type, top_poc);
                    sp.slice_data_size = slice_nal.data.len() as u32;
                    sp.slice_data_offset = 0;
                    sp.slice_data_flag = VA_SLICE_DATA_FLAG_ALL;
                    sp.slice_data_bit_offset = sh.header_bit_len as u16;
                    sp.first_mb_in_slice = sh.first_mb_in_slice as u16;
                    sp.slice_type = (sh.slice_type % 5) as u8;
                    sp.direct_spatial_mv_pred_flag = u8::from(sh.direct_spatial_mv_pred_flag);
                    sp.num_ref_idx_l0_active_minus1 = sh.num_ref_idx_l0_active_minus1 as u8;
                    sp.num_ref_idx_l1_active_minus1 = sh.num_ref_idx_l1_active_minus1 as u8;
                    sp.cabac_init_idc = sh.cabac_init_idc as u8;
                    sp.slice_qp_delta = sh.slice_qp_delta as i8;
                    sp.disable_deblocking_filter_idc = sh.disable_deblocking_filter_idc as u8;
                    sp.slice_alpha_c0_offset_div2 = sh.slice_alpha_c0_offset_div2 as i8;
                    sp.slice_beta_offset_div2 = sh.slice_beta_offset_div2 as i8;

                    if let Some(ref wt) = sh.weight_table {
                        sp.luma_log2_weight_denom = wt.luma_log2_denom as u8;
                        sp.chroma_log2_weight_denom = wt.chroma_log2_denom as u8;
                        if !wt.luma_l0.is_empty() {
                            sp.luma_weight_l0_flag = 1;
                            for (i, w) in wt.luma_l0.iter().take(32).enumerate() {
                                sp.luma_weight_l0[i] = w.weight as i16;
                                sp.luma_offset_l0[i] = w.offset as i16;
                            }
                        }
                        if !wt.chroma_l0.is_empty() {
                            sp.chroma_weight_l0_flag = 1;
                            for (i, cw) in wt.chroma_l0.iter().take(32).enumerate() {
                                sp.chroma_weight_l0[i] = [cw[0].weight as i16, cw[1].weight as i16];
                                sp.chroma_offset_l0[i] = [cw[0].offset as i16, cw[1].offset as i16];
                            }
                        }
                        if !wt.luma_l1.is_empty() {
                            sp.luma_weight_l1_flag = 1;
                            for (i, w) in wt.luma_l1.iter().take(32).enumerate() {
                                sp.luma_weight_l1[i] = w.weight as i16;
                                sp.luma_offset_l1[i] = w.offset as i16;
                            }
                        }
                        if !wt.chroma_l1.is_empty() {
                            sp.chroma_weight_l1_flag = 1;
                            for (i, cw) in wt.chroma_l1.iter().take(32).enumerate() {
                                sp.chroma_weight_l1[i] = [cw[0].weight as i16, cw[1].weight as i16];
                                sp.chroma_offset_l1[i] = [cw[0].offset as i16, cw[1].offset as i16];
                            }
                        }
                    }

                    let mut sp_buf: VABufferID = 0;
                    check_va(
                        (va.va_create_buffer)(
                            self.display,
                            self.context,
                            VASliceParameterBufferType,
                            std::mem::size_of::<VASliceParameterBufferH264>() as u32,
                            1,
                            &sp as *const _ as *const c_void,
                            &mut sp_buf,
                        ),
                        "vaCreateBuffer(SliceParameter)",
                    )?;
                    created_buffers.push(sp_buf);

                    let mut sd_buf: VABufferID = 0;
                    check_va(
                        (va.va_create_buffer)(
                            self.display,
                            self.context,
                            VASliceDataBufferType,
                            slice_nal.data.len() as u32,
                            1,
                            slice_nal.data.as_ptr() as *const c_void,
                            &mut sd_buf,
                        ),
                        "vaCreateBuffer(SliceData)",
                    )?;
                    created_buffers.push(sd_buf);

                    let mut slice_bufs = [sp_buf, sd_buf];
                    check_va(
                        (va.va_render_picture)(
                            self.display,
                            self.context,
                            slice_bufs.as_mut_ptr(),
                            2,
                        ),
                        "vaRenderPicture(slice)",
                    )?;
                }

                check_va(
                    (va.va_end_picture)(self.display, self.context),
                    "vaEndPicture",
                )?;
                picture_started = false;
                self.readback_surface(surface, timestamp_us, is_idr)
            })();

            if picture_started {
                (va.va_end_picture)(self.display, self.context);
            }
            for buffer in created_buffers {
                (va.va_destroy_buffer)(self.display, buffer);
            }

            let frame = match result {
                Ok(frame) => frame,
                Err(e) => {
                    self.surface_in_use[surf_idx] = false;
                    return Err(e);
                }
            };

            if nal_ref_idc != 0 {
                self.dpb.sliding_window_evict(&mut self.surface_in_use);
                self.dpb
                    .add_reference(surf_idx, first_sh.frame_num, top_poc, bottom_poc);
            } else {
                self.surface_in_use[surf_idx] = false;
            }

            self.current_surface_idx = surf_idx;
            self.frame_counter = self.frame_counter.saturating_add(1);
            Ok(Some(frame))
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

            if self.codec == VideoCodec::H264 {
                let mut slice_nals: Vec<&crate::NalUnit> = Vec::new();
                let mut dimensions_changed = false;

                for nal in &nals {
                    if nal.data.is_empty() {
                        continue;
                    }
                    match nal.data[0] & 0x1F {
                        7 => {
                            if let Ok(sps) = crate::h264_params::parse_sps(&nal.data[1..]) {
                                let new_profile = Self::select_h264_profile(sps.profile_idc);
                                if !self.surfaces_created && new_profile != self.h264_profile {
                                    // SAFETY: VA display/config are valid, context not created yet.
                                    unsafe {
                                        self.recreate_config(new_profile)?;
                                    }
                                }
                                if let Some(ref prev_sps) = self.sps
                                    && (prev_sps.cropped_width() != sps.cropped_width()
                                        || prev_sps.cropped_height() != sps.cropped_height())
                                {
                                    dimensions_changed = true;
                                }
                                self.sps = Some(sps);
                            }
                        }
                        8 => {
                            if let Ok(pps) = crate::h264_params::parse_pps(&nal.data[1..]) {
                                self.pps = Some(pps);
                            }
                        }
                        1 | 5 => slice_nals.push(nal),
                        _ => {}
                    }
                }

                if slice_nals.is_empty() {
                    return Ok(None);
                }

                // SAFETY: VA handles are valid for context/surface lifecycle operations.
                unsafe {
                    if dimensions_changed && self.surfaces_created {
                        self.destroy_surfaces_and_context();
                        for idx in self.dpb.clear() {
                            if idx < self.surface_in_use.len() {
                                self.surface_in_use[idx] = false;
                            }
                        }
                    }

                    if !self.surfaces_created {
                        let sps = self
                            .sps
                            .as_ref()
                            .ok_or_else(|| VideoError::Codec("no SPS".into()))?;
                        let w = sps.cropped_width() as u32;
                        let h = sps.cropped_height() as u32;
                        self.dpb.max_size = sps.max_num_ref_frames.min(16) as usize;
                        self.create_surfaces(w, h)?;
                    }

                    return self.decode_picture(&slice_nals, timestamp_us);
                }
            }

            let mut slice_data = Vec::new();
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                let is_param = matches!((nal.data[0] >> 1) & 0x3F, 32..=34);
                if !is_param {
                    slice_data.extend_from_slice(&nal.data);
                }
            }
            if slice_data.is_empty() {
                return Ok(None);
            }

            // SAFETY: VA context lifecycle and decode calls are validated by status checks.
            unsafe {
                if !self.surfaces_created {
                    self.create_surfaces(1920, 1080)?;
                }
                self.decode_legacy_slice(&slice_data, self.current_surface_idx, timestamp_us)
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
                            (va.va_destroy_context)(self.display, self.context);
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
    const MF_E_NOTACCEPTING: HRESULT = 0xC00D36B5_u32 as i32;
    #[allow(dead_code)]
    const MF_E_INVALIDMEDIATYPE: HRESULT = 0xC00D36B4_u32 as i32;
    #[allow(dead_code)]
    const E_NOTIMPL: HRESULT = 0x80004001_u32 as i32;

    // MFT category GUID for video decoders {d0033739-4f81-4293-868e-2f732875c515}
    // {d6c02d4b-6833-45b4-971a-05a4b04bab91}
    const MFT_CATEGORY_VIDEO_DECODER: GUID = [
        0x4b, 0x2d, 0xc0, 0xd6, 0x33, 0x68, 0xb4, 0x45, 0x97, 0x1a, 0x05, 0xa4, 0xb0, 0x4b, 0xab,
        0x91,
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

    // {d7388766-18fe-48c6-a177-ee894867c8c4}
    const MF_MT_MINIMUM_DISPLAY_APERTURE: GUID = [
        0x66, 0x87, 0x38, 0xd7, 0xfe, 0x18, 0xc6, 0x48, 0xa1, 0x77, 0xee, 0x89, 0x48, 0x67, 0xc8,
        0xc4,
    ];

    // MFT_MESSAGE constants
    const MFT_MESSAGE_COMMAND_FLUSH: u32 = 0x0;
    const MFT_MESSAGE_COMMAND_DRAIN: u32 = 0x1;
    const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x10000000;
    const MFT_MESSAGE_NOTIFY_START_OF_STREAM: u32 = 0x10000003;

    const MFT_OUTPUT_STREAM_PROVIDES_SAMPLES: u32 = 0x100;

    // DXVA / D3D11 constants
    const MFT_MESSAGE_SET_D3D_MANAGER: u32 = 0x2;
    const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
    const D3D11_CREATE_DEVICE_VIDEO_SUPPORT: u32 = 0x800;
    const D3D11_SDK_VERSION: u32 = 7;
    const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
    const D3D11_USAGE_STAGING: u32 = 3;
    const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
    const D3D11_MAP_READ: u32 = 1;
    const DXGI_FORMAT_NV12: u32 = 103;

    // IID_IMFDXGIDeviceManager {eb533d5d-2db6-40f8-97a9-494692014f07}
    #[allow(dead_code)]
    const IID_IMFDXGIDeviceManager: GUID = [
        0x5d, 0x3d, 0x53, 0xeb, 0xb6, 0x2d, 0xf8, 0x40, 0x97, 0xa9, 0x49, 0x46, 0x92, 0x01, 0x4f,
        0x07,
    ];

    // IID_IMFDXGIBuffer {e7174cfa-1c9e-48b1-8866-626226bfc258}
    const IID_IMFDXGIBuffer: GUID = [
        0xfa, 0x4c, 0x17, 0xe7, 0x9e, 0x1c, 0xb1, 0x48, 0x88, 0x66, 0x62, 0x62, 0x26, 0xbf, 0xc2,
        0x58,
    ];

    // IID_ID3D11Texture2D {6f15aaf2-d208-4e89-9ab4-489535d34f9c}
    const IID_ID3D11Texture2D: GUID = [
        0xf2, 0xaa, 0x15, 0x6f, 0x08, 0xd2, 0x89, 0x4e, 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f,
        0x9c,
    ];

    // IID_ID3D10Multithread {9B7E4E00-342C-4106-A19F-4F2704F689F0}
    const IID_ID3D10Multithread: GUID = [
        0x00, 0x4e, 0x7e, 0x9b, 0x2c, 0x34, 0x06, 0x41, 0xa1, 0x9f, 0x4f, 0x27, 0x04, 0xf6, 0x89,
        0xf0,
    ];

    // ── Extern function bindings ──────────────────────────────────────

    #[link(name = "mfplat")]
    unsafe extern "system" {
        fn MFStartup(version: u32, flags: u32) -> HRESULT;
        fn MFShutdown() -> HRESULT;
        fn MFCreateMediaType(media_type: *mut *mut c_void) -> HRESULT;
    }

    #[link(name = "mfplat")]
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

    #[link(name = "mfplat")]
    unsafe extern "system" {
        fn MFCreateDXGIDeviceManager(
            reset_token: *mut u32,
            pp_device_manager: *mut *mut c_void,
        ) -> HRESULT;
    }

    #[link(name = "d3d11")]
    unsafe extern "system" {
        fn D3D11CreateDevice(
            p_adapter: *mut c_void,
            driver_type: u32,
            software: *mut c_void,
            flags: u32,
            p_feature_levels: *const u32,
            feature_levels: u32,
            sdk_version: u32,
            pp_device: *mut *mut c_void,
            p_feature_level: *mut u32,
            pp_immediate_context: *mut *mut c_void,
        ) -> HRESULT;
    }

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn CoInitializeEx(pvReserved: *mut c_void, dwCoInit: u32) -> HRESULT;
        fn CoTaskMemFree(pv: *mut c_void);
        fn CoCreateInstance(
            rclsid: *const GUID,
            p_unk_outer: *mut c_void,
            dw_cls_context: u32,
            riid: *const GUID,
            ppv: *mut *mut c_void,
        ) -> HRESULT;
    }

    const CLSCTX_INPROC_SERVER: u32 = 0x1;

    // {62CE7E72-4C71-4D20-B15D-452831A87D9D}
    const CLSID_CMSH264DecoderMFT: GUID = [
        0x72, 0x7e, 0xce, 0x62, 0x71, 0x4c, 0x20, 0x4d, 0xb1, 0x5d, 0x45, 0x28, 0x31, 0xa8, 0x7d,
        0x9d,
    ];

    // {420A51A3-D605-430C-B4FC-45274FA6C562}
    const CLSID_CMSHEVCDecoderMFT: GUID = [
        0xa3, 0x51, 0x0a, 0x42, 0x05, 0xd6, 0x0c, 0x43, 0xb4, 0xfc, 0x45, 0x27, 0x4f, 0xa6, 0xc5,
        0x62,
    ];

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

    #[repr(C)]
    struct D3D11_TEXTURE2D_DESC {
        width: u32,
        height: u32,
        mip_levels: u32,
        array_size: u32,
        format: u32,
        sample_count: u32,
        sample_quality: u32,
        usage: u32,
        bind_flags: u32,
        cpu_access_flags: u32,
        misc_flags: u32,
    }

    #[repr(C)]
    struct D3D11_MAPPED_SUBRESOURCE {
        p_data: *mut c_void,
        row_pitch: u32,
        _depth_pitch: u32,
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

    /// IMFTransform::GetInputAvailableType (vtable 13)
    unsafe fn transform_get_input_available_type(
        transform: *mut c_void,
        stream_id: u32,
        type_idx: u32,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(13));
        method(transform, stream_id, type_idx, out)
    }

    /// IMFTransform::GetOutputAvailableType (vtable 14)
    unsafe fn transform_get_output_available_type(
        transform: *mut c_void,
        stream_id: u32,
        type_idx: u32,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(transform as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(14));
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
        let method: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(18));
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
    unsafe fn attributes_get_guid(attrs: *mut c_void, key: *const GUID, out: *mut GUID) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *const GUID, *mut GUID) -> HRESULT =
            std::mem::transmute(*vtable.add(10));
        method(attrs, key, out)
    }

    /// IMFAttributes::GetAllocatedBlob (vtable 16)
    unsafe fn attributes_get_blob(
        attrs: *mut c_void,
        key: *const GUID,
        buf: *mut *mut u8,
        len: *mut u32,
    ) -> HRESULT {
        let vtable = *(attrs as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const GUID,
            *mut *mut u8,
            *mut u32,
        ) -> HRESULT = std::mem::transmute(*vtable.add(16));
        method(attrs, key, buf, len)
    }

    /// IMFAttributes::SetUINT32 (vtable 21)
    unsafe fn attributes_set_uint32(attrs: *mut c_void, key: *const GUID, value: u32) -> HRESULT {
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
        let method: unsafe extern "system" fn(*mut c_void, *const GUID, *const GUID) -> HRESULT =
            std::mem::transmute(*vtable.add(24));
        method(attrs, key, value)
    }

    // IID for IMF2DBuffer {7DC9D5F9-9ED9-44ec-9BBF-0600BB589FBB}
    const IID_IMF2DBuffer: GUID = [
        0xf9, 0xd5, 0xc9, 0x7d, 0x9d, 0x9e, 0xec, 0x44, 0x9b, 0xbf, 0x06, 0x00, 0xbb, 0x58, 0x9f,
        0xbb,
    ];

    /// IUnknown::QueryInterface (vtable 0)
    unsafe fn com_query_interface(
        obj: *mut c_void,
        iid: *const GUID,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(obj as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const GUID,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(0));
        method(obj, iid, out)
    }

    /// IMF2DBuffer::Lock2D (vtable 3) — returns scanline pointer and pitch
    unsafe fn buffer_2d_lock(
        buf2d: *mut c_void,
        scanline0: *mut *mut u8,
        pitch: *mut i32,
    ) -> HRESULT {
        let vtable = *(buf2d as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut *mut u8, *mut i32) -> HRESULT =
            std::mem::transmute(*vtable.add(3));
        method(buf2d, scanline0, pitch)
    }

    /// IMF2DBuffer::Unlock2D (vtable 4)
    unsafe fn buffer_2d_unlock(buf2d: *mut c_void) -> HRESULT {
        let vtable = *(buf2d as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(4));
        method(buf2d)
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

    /// IMFSample::GetBufferByIndex (vtable 40)
    unsafe fn sample_get_buffer_by_index(
        sample: *mut c_void,
        index: u32,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(sample as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(40));
        method(sample, index, out)
    }

    /// IMFSample::AddBuffer (vtable 42)
    unsafe fn sample_add_buffer(sample: *mut c_void, buffer: *mut c_void) -> HRESULT {
        let vtable = *(sample as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT =
            std::mem::transmute(*vtable.add(42));
        method(sample, buffer)
    }

    // ── DXVA vtable helpers ─────────────────────────────────────────

    /// IMFDXGIDeviceManager::ResetDevice (vtable 7)
    unsafe fn dxgi_manager_reset_device(
        manager: *mut c_void,
        device: *mut c_void,
        reset_token: u32,
    ) -> HRESULT {
        let vtable = *(manager as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> HRESULT =
            std::mem::transmute(*vtable.add(7));
        method(manager, device, reset_token)
    }

    /// IMFDXGIBuffer::GetResource (vtable 3)
    unsafe fn dxgi_buffer_get_resource(
        buffer: *mut c_void,
        riid: *const GUID,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(buffer as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const GUID,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(3));
        method(buffer, riid, out)
    }

    /// IMFDXGIBuffer::GetSubresourceIndex (vtable 4)
    unsafe fn dxgi_buffer_get_subresource_index(buffer: *mut c_void, out: *mut u32) -> HRESULT {
        let vtable = *(buffer as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT =
            std::mem::transmute(*vtable.add(4));
        method(buffer, out)
    }

    /// ID3D11Device::CreateTexture2D (vtable 5)
    unsafe fn d3d11_create_texture_2d(
        device: *mut c_void,
        desc: *const D3D11_TEXTURE2D_DESC,
        initial_data: *const c_void,
        out: *mut *mut c_void,
    ) -> HRESULT {
        let vtable = *(device as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *const D3D11_TEXTURE2D_DESC,
            *const c_void,
            *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(*vtable.add(5));
        method(device, desc, initial_data, out)
    }

    /// ID3D11DeviceContext::Map (vtable 14)
    unsafe fn d3d11_context_map(
        context: *mut c_void,
        resource: *mut c_void,
        subresource: u32,
        map_type: u32,
        map_flags: u32,
        mapped: *mut D3D11_MAPPED_SUBRESOURCE,
    ) -> HRESULT {
        let vtable = *(context as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            u32,
            *mut D3D11_MAPPED_SUBRESOURCE,
        ) -> HRESULT = std::mem::transmute(*vtable.add(14));
        method(context, resource, subresource, map_type, map_flags, mapped)
    }

    /// ID3D11DeviceContext::Unmap (vtable 15)
    unsafe fn d3d11_context_unmap(context: *mut c_void, resource: *mut c_void, subresource: u32) {
        let vtable = *(context as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) =
            std::mem::transmute(*vtable.add(15));
        method(context, resource, subresource)
    }

    /// ID3D11DeviceContext::CopySubresourceRegion (vtable 46)
    #[allow(clippy::too_many_arguments)]
    unsafe fn d3d11_context_copy_subresource_region(
        context: *mut c_void,
        dst: *mut c_void,
        dst_subresource: u32,
        dst_x: u32,
        dst_y: u32,
        dst_z: u32,
        src: *mut c_void,
        src_subresource: u32,
        src_box: *const c_void,
    ) {
        let vtable = *(context as *const *const *const c_void);
        let method: unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            u32,
            u32,
            *mut c_void,
            u32,
            *const c_void,
        ) = std::mem::transmute(*vtable.add(46));
        method(
            context,
            dst,
            dst_subresource,
            dst_x,
            dst_y,
            dst_z,
            src,
            src_subresource,
            src_box,
        )
    }

    /// ID3D10Multithread::SetMultithreadProtected (vtable 5)
    unsafe fn d3d10_multithread_set_protected(mt: *mut c_void, protect: i32) -> i32 {
        let vtable = *(mt as *const *const *const c_void);
        let method: unsafe extern "system" fn(*mut c_void, i32) -> i32 =
            std::mem::transmute(*vtable.add(5));
        method(mt, protect)
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
        stride: u32,
        coded_height: u32,
        crop_x: u32,
        crop_y: u32,
        transform: *mut c_void,
        provides_samples: bool,
        output_buf_size: u32,
        cached_sample: *mut c_void,
        cached_buffer: *mut c_void,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
        dxva_enabled: bool,
        d3d11_device: *mut c_void,
        d3d11_context: *mut c_void,
        dxgi_manager: *mut c_void,
        _reset_token: u32,
        staging_texture: *mut c_void,
        staging_width: u32,
        staging_height: u32,
    }

    impl MediaFoundationDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            // SAFETY: (category 1) MF startup/enum/activate sequence; HRESULT checked
            // at each step; falls back to software on any failure.
            unsafe { Self::init_mft(codec) }
        }

        unsafe fn init_mft(codec: VideoCodec) -> Result<Self, VideoError> {
            let _ = CoInitializeEx(ptr::null_mut(), 0x2); // COINIT_APARTMENTTHREADED
            let hr = MFStartup(0x00020070, 0);
            if hr != S_OK {
                return Ok(Self::with_sw_fallback(codec));
            }

            let subtype = match codec {
                VideoCodec::H264 => MFVideoFormat_H264,
                VideoCodec::H265 => MFVideoFormat_HEVC,
                _ => return Err(VideoError::Codec("MF: unsupported codec".into())),
            };

            let clsid = match codec {
                VideoCodec::H264 => &CLSID_CMSH264DecoderMFT,
                _ => &CLSID_CMSHEVCDecoderMFT,
            };

            // DXVA: create D3D11 device + DXGI device manager
            let (d3d11_device, d3d11_context, dxgi_manager, reset_token) = Self::try_init_dxva();

            let mut transform: *mut c_void = ptr::null_mut();
            let hr = CoCreateInstance(
                clsid,
                ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IMF_TRANSFORM,
                &mut transform,
            );
            if hr != S_OK || transform.is_null() {
                eprintln!("[MF] CoCreateInstance failed: hr={hr:#X}");
                Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }
            eprintln!("[MF] CoCreateInstance OK: {codec:?} transform={transform:?}");

            // Set D3D manager on MFT BEFORE type negotiation
            let dxva_enabled = if !dxgi_manager.is_null() {
                let hr = transform_process_message(
                    transform,
                    MFT_MESSAGE_SET_D3D_MANAGER,
                    dxgi_manager as u64,
                );
                if hr == S_OK {
                    eprintln!("[MF] DXVA: SET_D3D_MANAGER OK");
                    true
                } else {
                    eprintln!(
                        "[MF] DXVA: SET_D3D_MANAGER failed: hr={hr:#X}, falling back to software"
                    );
                    Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                    false
                }
            } else {
                false
            };

            let mut input_type: *mut c_void = ptr::null_mut();
            let mut found_input = false;
            for idx in 0..64u32 {
                let mut candidate: *mut c_void = ptr::null_mut();
                let hr = transform_get_input_available_type(transform, 0, idx, &mut candidate);
                if hr != S_OK || candidate.is_null() {
                    break;
                }
                let mut sub: GUID = [0u8; 16];
                if attributes_get_guid(candidate, &MF_MT_SUBTYPE, &mut sub) == S_OK {
                    eprintln!("[MF] input available[{idx}]: sub={sub:02x?}");
                    if sub == subtype {
                        input_type = candidate;
                        found_input = true;
                        break;
                    }
                }
                com_release(candidate);
            }
            if !found_input || input_type.is_null() {
                eprintln!("[MF] No matching input type for {codec:?}");
                if !dxva_enabled {
                    Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                }
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            let hr = transform_set_input_type(transform, 0, input_type, 0);
            com_release(input_type);
            if hr != S_OK {
                eprintln!("[MF] SetInputType failed: hr={hr:#X}");
                if !dxva_enabled {
                    Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                }
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }
            eprintln!("[MF] SetInputType OK");

            let mut nv12_type: *mut c_void = ptr::null_mut();
            for idx in 0..64u32 {
                let mut candidate: *mut c_void = ptr::null_mut();
                let hr = transform_get_output_available_type(transform, 0, idx, &mut candidate);
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
                eprintln!("[MF] No NV12 output type available");
                if !dxva_enabled {
                    Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                }
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

            let hr = transform_set_output_type(transform, 0, nv12_type, 0);
            eprintln!("[MF] SetOutputType(NV12): hr={hr:#X}");

            let mut width: u32 = 0;
            let mut height: u32 = 0;
            let mut frame_size: u64 = 0;
            if attributes_get_uint64(nv12_type, &MF_MT_FRAME_SIZE, &mut frame_size) == S_OK {
                width = (frame_size >> 32) as u32;
                height = frame_size as u32;
            }
            com_release(nv12_type);

            if hr != S_OK {
                if !dxva_enabled {
                    Self::cleanup_dxva(d3d11_device, d3d11_context, dxgi_manager);
                }
                com_release(transform);
                MFShutdown();
                return Ok(Self::with_sw_fallback(codec));
            }

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

            transform_process_message(transform, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
            transform_process_message(transform, MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);

            eprintln!(
                "[MF] Decoder ready: {codec:?} {width}x{height} dxva={dxva_enabled} provides_samples={provides_samples} buf_size={output_buf_size}"
            );

            Ok(MediaFoundationDecoder {
                codec,
                initialized: true,
                width,
                height,
                stride: 0,
                coded_height: 0,
                crop_x: 0,
                crop_y: 0,
                transform,
                provides_samples,
                output_buf_size,
                cached_sample: ptr::null_mut(),
                cached_buffer: ptr::null_mut(),
                sw_fallback: None,
                dxva_enabled,
                d3d11_device: if dxva_enabled {
                    d3d11_device
                } else {
                    ptr::null_mut()
                },
                d3d11_context: if dxva_enabled {
                    d3d11_context
                } else {
                    ptr::null_mut()
                },
                dxgi_manager: if dxva_enabled {
                    dxgi_manager
                } else {
                    ptr::null_mut()
                },
                _reset_token: if dxva_enabled { reset_token } else { 0 },
                staging_texture: ptr::null_mut(),
                staging_width: 0,
                staging_height: 0,
            })
        }

        /// Attempt D3D11 device + DXGI manager creation for DXVA.
        /// Returns (device, context, manager, token) — all null on failure.
        unsafe fn try_init_dxva() -> (*mut c_void, *mut c_void, *mut c_void, u32) {
            let null = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), 0u32);

            let feature_levels = [D3D_FEATURE_LEVEL_11_0];
            let mut device: *mut c_void = ptr::null_mut();
            let mut context: *mut c_void = ptr::null_mut();
            let mut _feature_level: u32 = 0;

            let hr = D3D11CreateDevice(
                ptr::null_mut(),
                D3D_DRIVER_TYPE_HARDWARE,
                ptr::null_mut(),
                D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                feature_levels.as_ptr(),
                feature_levels.len() as u32,
                D3D11_SDK_VERSION,
                &mut device,
                &mut _feature_level,
                &mut context,
            );
            if hr != S_OK || device.is_null() || context.is_null() {
                eprintln!("[MF] DXVA: D3D11CreateDevice failed: hr={hr:#X}");
                if !device.is_null() {
                    com_release(device);
                }
                if !context.is_null() {
                    com_release(context);
                }
                return null;
            }
            eprintln!("[MF] DXVA: D3D11 device created");

            // Enable multithread protection (MF uses worker threads on this device)
            let mut mt: *mut c_void = ptr::null_mut();
            if com_query_interface(device, &IID_ID3D10Multithread, &mut mt) == S_OK && !mt.is_null()
            {
                d3d10_multithread_set_protected(mt, 1);
                com_release(mt);
            }

            let mut reset_token: u32 = 0;
            let mut manager: *mut c_void = ptr::null_mut();
            let hr = MFCreateDXGIDeviceManager(&mut reset_token, &mut manager);
            if hr != S_OK || manager.is_null() {
                eprintln!("[MF] DXVA: MFCreateDXGIDeviceManager failed: hr={hr:#X}");
                com_release(device);
                com_release(context);
                return null;
            }

            let hr = dxgi_manager_reset_device(manager, device, reset_token);
            if hr != S_OK {
                eprintln!("[MF] DXVA: ResetDevice failed: hr={hr:#X}");
                com_release(manager);
                com_release(device);
                com_release(context);
                return null;
            }
            eprintln!("[MF] DXVA: device manager ready");

            (device, context, manager, reset_token)
        }

        unsafe fn cleanup_dxva(device: *mut c_void, context: *mut c_void, manager: *mut c_void) {
            if !manager.is_null() {
                com_release(manager);
            }
            if !context.is_null() {
                com_release(context);
            }
            if !device.is_null() {
                com_release(device);
            }
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
                stride: 0,
                coded_height: 0,
                crop_x: 0,
                crop_y: 0,
                transform: ptr::null_mut(),
                provides_samples: false,
                output_buf_size: 0,
                cached_sample: ptr::null_mut(),
                cached_buffer: ptr::null_mut(),
                sw_fallback: Some(sw),
                dxva_enabled: false,
                d3d11_device: ptr::null_mut(),
                d3d11_context: ptr::null_mut(),
                dxgi_manager: ptr::null_mut(),
                _reset_token: 0,
                staging_texture: ptr::null_mut(),
                staging_width: 0,
                staging_height: 0,
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

            // ProcessInput — if MFT is full, drain output first then retry
            let mut hr = transform_process_input(self.transform, 0, sample, 0);
            if hr == MF_E_NOTACCEPTING {
                let frame = self.try_drain_output(timestamp_us)?;
                if frame.is_some() {
                    hr = transform_process_input(self.transform, 0, sample, 0);
                    if hr == S_OK {
                        com_release(sample);
                        return Ok(frame);
                    }
                }
            }
            com_release(sample);
            if hr != S_OK {
                eprintln!(
                    "[MF] ProcessInput failed: hr={hr:#X} data_len={}",
                    data.len()
                );
                return Err(VideoError::Codec(format!(
                    "MF: ProcessInput failed: {hr:#X}"
                )));
            }

            self.try_drain_output(timestamp_us)
        }

        unsafe fn renegotiate_output(&mut self) {
            for idx in 0..64u32 {
                let mut candidate: *mut c_void = ptr::null_mut();
                let hr =
                    transform_get_output_available_type(self.transform, 0, idx, &mut candidate);
                if hr != S_OK || candidate.is_null() {
                    break;
                }
                let mut sub: GUID = [0u8; 16];
                if attributes_get_guid(candidate, &MF_MT_SUBTYPE, &mut sub) == S_OK
                    && sub == MFVideoFormat_NV12
                {
                    let hr = transform_set_output_type(self.transform, 0, candidate, 0);
                    if hr == S_OK {
                        let mut frame_size: u64 = 0;
                        if attributes_get_uint64(candidate, &MF_MT_FRAME_SIZE, &mut frame_size)
                            == S_OK
                        {
                            self.width = (frame_size >> 32) as u32;
                            self.height = frame_size as u32;
                        }
                        eprintln!("[MF] Re-negotiated output: {}x{}", self.width, self.height);
                    }
                    com_release(candidate);
                    return;
                }
                com_release(candidate);
            }
            eprintln!("[MF] Re-negotiation failed: no NV12 output type");
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

            let pre_alloc = if !self.provides_samples {
                if self.cached_sample.is_null() {
                    let mut out_sample: *mut c_void = ptr::null_mut();
                    if MFCreateSample(&mut out_sample) != S_OK || out_sample.is_null() {
                        return Ok(None);
                    }
                    let buf_size = if self.output_buf_size > 0 {
                        self.output_buf_size
                    } else {
                        4096 * 2160 * 3 / 2
                    };
                    let mut out_buf: *mut c_void = ptr::null_mut();
                    if MFCreateMemoryBuffer(buf_size, &mut out_buf) != S_OK || out_buf.is_null() {
                        com_release(out_sample);
                        return Ok(None);
                    }
                    sample_add_buffer(out_sample, out_buf);
                    self.cached_sample = out_sample;
                    self.cached_buffer = out_buf;
                }
                media_buffer_set_current_length(self.cached_buffer, 0);
                output_buf.sample = self.cached_sample;
                self.cached_sample
            } else {
                ptr::null_mut()
            };

            let mut proc_status: u32 = 0;
            let hr =
                transform_process_output(self.transform, 0, 1, &mut output_buf, &mut proc_status);

            if !output_buf.events.is_null() {
                com_release(output_buf.events);
            }

            const MF_E_TRANSFORM_STREAM_CHANGE: HRESULT = 0xC00D6D61_u32 as i32;
            if hr == MF_E_TRANSFORM_STREAM_CHANGE {
                if !pre_alloc.is_null() && pre_alloc != self.cached_sample {
                    com_release(pre_alloc);
                }
                if !self.cached_buffer.is_null() {
                    com_release(self.cached_buffer);
                    self.cached_buffer = ptr::null_mut();
                }
                if !self.cached_sample.is_null() {
                    com_release(self.cached_sample);
                    self.cached_sample = ptr::null_mut();
                }
                self.renegotiate_output();
                self.width = 0;
                self.height = 0;
                self.stride = 0;
                self.coded_height = 0;
                self.crop_x = 0;
                self.crop_y = 0;
                eprintln!("[MF] Stream change, will re-detect dimensions on next frame");
                return Ok(None);
            }
            if hr != S_OK && hr != MF_E_TRANSFORM_NEED_MORE_INPUT {
                if !pre_alloc.is_null() {
                    com_release(pre_alloc);
                }
                return Err(VideoError::Codec(format!(
                    "MF: ProcessOutput failed: {hr:#X}"
                )));
            }
            if hr == MF_E_TRANSFORM_NEED_MORE_INPUT {
                return Ok(None);
            }
            if hr != S_OK {
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

            let result = if self.dxva_enabled {
                self.extract_dxva_frame(out_sample, timestamp_us)
            } else {
                self.extract_software_frame(out_sample, timestamp_us)
            };

            // Release MFT-provided sample (DXVA: provides_samples=true)
            if self.provides_samples && out_sample != pre_alloc && !out_sample.is_null() {
                com_release(out_sample);
            }

            result
        }

        unsafe fn extract_dxva_frame(
            &mut self,
            out_sample: *mut c_void,
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let mut buffer: *mut c_void = ptr::null_mut();
            let hr = sample_get_buffer_by_index(out_sample, 0, &mut buffer);
            if hr != S_OK || buffer.is_null() {
                return Err(VideoError::Codec(format!(
                    "MF DXVA: GetBufferByIndex failed: {hr:#X}"
                )));
            }

            let mut dxgi_buf: *mut c_void = ptr::null_mut();
            let hr = com_query_interface(buffer, &IID_IMFDXGIBuffer, &mut dxgi_buf);
            com_release(buffer);
            if hr != S_OK || dxgi_buf.is_null() {
                return Err(VideoError::Codec(format!(
                    "MF DXVA: QI IMFDXGIBuffer failed: {hr:#X}"
                )));
            }

            let mut texture: *mut c_void = ptr::null_mut();
            dxgi_buffer_get_resource(dxgi_buf, &IID_ID3D11Texture2D, &mut texture);
            let mut subresource: u32 = 0;
            dxgi_buffer_get_subresource_index(dxgi_buf, &mut subresource);
            com_release(dxgi_buf);

            if texture.is_null() {
                return Err(VideoError::Codec(
                    "MF DXVA: GetResource returned null".into(),
                ));
            }

            if self.width == 0 || self.height == 0 || self.stride == 0 {
                self.resolve_output_dimensions();
            }

            if self.width == 0 || self.height == 0 {
                com_release(texture);
                return Err(VideoError::Codec(
                    "MF DXVA: cannot determine output dimensions".into(),
                ));
            }

            let coded_w = if self.stride > 0 {
                self.stride
            } else {
                self.width
            };
            let coded_h = if self.coded_height > 0 {
                self.coded_height
            } else {
                self.height
            };
            if self.staging_texture.is_null()
                || self.staging_width != coded_w
                || self.staging_height != coded_h
            {
                if !self.staging_texture.is_null() {
                    com_release(self.staging_texture);
                }
                let desc = D3D11_TEXTURE2D_DESC {
                    width: coded_w,
                    height: coded_h,
                    mip_levels: 1,
                    array_size: 1,
                    format: DXGI_FORMAT_NV12,
                    sample_count: 1,
                    sample_quality: 0,
                    usage: D3D11_USAGE_STAGING,
                    bind_flags: 0,
                    cpu_access_flags: D3D11_CPU_ACCESS_READ,
                    misc_flags: 0,
                };
                let mut staging: *mut c_void = ptr::null_mut();
                let hr =
                    d3d11_create_texture_2d(self.d3d11_device, &desc, ptr::null(), &mut staging);
                if hr != S_OK || staging.is_null() {
                    com_release(texture);
                    return Err(VideoError::Codec(format!(
                        "MF DXVA: CreateTexture2D staging failed: {hr:#X}"
                    )));
                }
                self.staging_texture = staging;
                self.staging_width = coded_w;
                self.staging_height = coded_h;
                eprintln!("[MF] DXVA staging texture: {coded_w}x{coded_h}");
            }

            d3d11_context_copy_subresource_region(
                self.d3d11_context,
                self.staging_texture,
                0,
                0,
                0,
                0,
                texture,
                subresource,
                ptr::null(),
            );
            com_release(texture);

            let mut mapped = D3D11_MAPPED_SUBRESOURCE {
                p_data: ptr::null_mut(),
                row_pitch: 0,
                _depth_pitch: 0,
            };
            let hr = d3d11_context_map(
                self.d3d11_context,
                self.staging_texture,
                0,
                D3D11_MAP_READ,
                0,
                &mut mapped,
            );
            if hr != S_OK || mapped.p_data.is_null() {
                return Err(VideoError::Codec(format!(
                    "MF DXVA: Map staging failed: {hr:#X}"
                )));
            }

            let stride = mapped.row_pitch as usize;
            if self.stride == 0 {
                self.stride = mapped.row_pitch;
            }

            let w = self.width as usize;
            let h = self.height as usize;
            let crop_x = self.crop_x as usize;
            let crop_y = self.crop_y as usize;
            let coded_h_usize = coded_h as usize;

            let nv12_ptr = mapped.p_data as *const u8;
            let y_start = nv12_ptr.add(crop_y * stride + crop_x);
            let uv_start =
                nv12_ptr.add(stride * coded_h_usize + (crop_y / 2) * stride + (crop_x & !1));

            let mut rgb = vec![0u8; w * h * 3];
            super::nv12_to_rgb8(y_start, stride, uv_start, stride, w, h, &mut rgb);

            d3d11_context_unmap(self.d3d11_context, self.staging_texture, 0);

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

        unsafe fn extract_software_frame(
            &mut self,
            out_sample: *mut c_void,
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            let mut contig_buf: *mut c_void = ptr::null_mut();
            let hr = sample_convert_to_contiguous_buffer(out_sample, &mut contig_buf);
            if hr != S_OK || contig_buf.is_null() {
                return Err(VideoError::Codec(format!(
                    "MF: ConvertToContiguousBuffer failed: {hr:#X}"
                )));
            }

            let mut nv12_ptr: *mut u8 = ptr::null_mut();
            let mut max_len: u32 = 0;
            let mut nv12_len: u32 = 0;
            let hr = media_buffer_lock(contig_buf, &mut nv12_ptr, &mut max_len, &mut nv12_len);
            if hr != S_OK {
                if contig_buf != self.cached_buffer {
                    com_release(contig_buf);
                }
                return Err(VideoError::Codec(format!(
                    "MF: Lock output buffer failed: {hr:#X}"
                )));
            }

            if self.width == 0 || self.height == 0 {
                self.resolve_output_dimensions();
            }

            if self.width == 0 || self.height == 0 {
                media_buffer_unlock(contig_buf);
                if contig_buf != self.cached_buffer {
                    com_release(contig_buf);
                }
                return Err(VideoError::Codec(
                    "MF: cannot determine output dimensions".into(),
                ));
            }

            if self.stride == 0 {
                let nv12_total = nv12_len as usize;
                let coded_w = self.width;
                let coded_h = self.height;
                self.stride = if nv12_total == coded_w as usize * coded_h as usize * 3 / 2 {
                    coded_w
                } else {
                    (nv12_total * 2 / (coded_h as usize * 3)) as u32
                };
                self.coded_height = coded_h;
                eprintln!(
                    "[MF] Resolved: {}x{} stride={} nv12_len={nv12_len}",
                    self.width, self.height, self.stride
                );
            }

            if self.stride == 0 {
                media_buffer_unlock(contig_buf);
                if contig_buf != self.cached_buffer {
                    com_release(contig_buf);
                }
                return Err(VideoError::Codec("MF: cannot determine stride".into()));
            }

            let w = self.width as usize;
            let h = self.height as usize;
            let stride = self.stride as usize;
            let crop_x = self.crop_x as usize;
            let crop_y = self.crop_y as usize;
            let coded_h = self.coded_height as usize;

            let y_start = nv12_ptr.add(crop_y * stride + crop_x);
            let uv_start = nv12_ptr.add(stride * coded_h + (crop_y / 2) * stride + (crop_x & !1));

            let mut rgb = vec![0u8; w * h * 3];
            super::nv12_to_rgb8(y_start, stride, uv_start, stride, w, h, &mut rgb);

            media_buffer_unlock(contig_buf);
            if contig_buf != self.cached_buffer {
                com_release(contig_buf);
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

        unsafe fn resolve_output_dimensions(&mut self) {
            let mut out_type: *mut c_void = ptr::null_mut();
            if transform_get_output_current_type(self.transform, 0, &mut out_type) != S_OK
                || out_type.is_null()
            {
                return;
            }
            let mut frame_size: u64 = 0;
            if attributes_get_uint64(out_type, &MF_MT_FRAME_SIZE, &mut frame_size) == S_OK {
                let coded_w = (frame_size >> 32) as u32;
                let coded_h = frame_size as u32;
                if coded_w > 0 && coded_h > 0 {
                    if self.width == 0 {
                        self.width = coded_w;
                    }
                    if self.height == 0 {
                        self.height = coded_h;
                    }
                    if self.coded_height == 0 {
                        self.coded_height = coded_h;
                    }

                    // MFVideoArea aperture crop
                    let mut aperture_blob: *mut u8 = ptr::null_mut();
                    let mut aperture_len: u32 = 0;
                    if attributes_get_blob(
                        out_type,
                        &MF_MT_MINIMUM_DISPLAY_APERTURE,
                        &mut aperture_blob,
                        &mut aperture_len,
                    ) == S_OK
                        && !aperture_blob.is_null()
                        && aperture_len >= 16
                    {
                        let offset_x = *(aperture_blob.add(2) as *const i16) as u32;
                        let offset_y = *(aperture_blob.add(6) as *const i16) as u32;
                        let display_w = *(aperture_blob.add(8) as *const i32) as u32;
                        let display_h = *(aperture_blob.add(12) as *const i32) as u32;
                        if display_w > 0
                            && display_w <= coded_w
                            && display_h > 0
                            && display_h <= coded_h
                        {
                            self.width = display_w;
                            self.height = display_h;
                            self.crop_x = offset_x;
                            self.crop_y = offset_y;
                        }
                        CoTaskMemFree(aperture_blob as *mut c_void);
                    }
                    eprintln!(
                        "[MF] Resolved dimensions: {}x{} coded={}x{}",
                        self.width, self.height, coded_w, coded_h
                    );
                }
            }
            com_release(out_type);
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
                unsafe {
                    if !self.transform.is_null() {
                        if self.dxva_enabled {
                            transform_process_message(
                                self.transform,
                                MFT_MESSAGE_SET_D3D_MANAGER,
                                0,
                            );
                        }
                        transform_process_message(self.transform, MFT_MESSAGE_COMMAND_FLUSH, 0);
                        com_release(self.transform);
                    }
                    if !self.cached_buffer.is_null() {
                        com_release(self.cached_buffer);
                    }
                    if !self.cached_sample.is_null() {
                        com_release(self.cached_sample);
                    }
                    if !self.staging_texture.is_null() {
                        com_release(self.staging_texture);
                    }
                    if !self.dxgi_manager.is_null() {
                        com_release(self.dxgi_manager);
                    }
                    if !self.d3d11_context.is_null() {
                        com_release(self.d3d11_context);
                    }
                    if !self.d3d11_device.is_null() {
                        com_release(self.d3d11_device);
                    }
                    MFShutdown();
                }
            }
        }
    }

    unsafe impl Send for MediaFoundationDecoder {}

    pub fn probe_hevc() -> bool {
        unsafe {
            let _ = CoInitializeEx(ptr::null_mut(), 0x2);
            let mut transform: *mut c_void = ptr::null_mut();
            let hr = CoCreateInstance(
                &CLSID_CMSHEVCDecoderMFT,
                ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IMF_TRANSFORM,
                &mut transform,
            );
            let available = hr == S_OK && !transform.is_null();
            if !transform.is_null() {
                com_release(transform);
            }
            available
        }
    }
}

/// Check whether the host can hardware-decode H.265/HEVC via a working backend.
///
/// On Windows: probes whether the Media Foundation HEVC decoder MFT can be
/// instantiated (requires HEVC Video Extensions from Microsoft Store).
/// On macOS: VideoToolbox always supports HEVC.
/// On Linux: VA-API implementation is incomplete (missing parameter buffers),
/// NVDEC is not yet validated. Returns false until a working backend exists.
#[allow(unreachable_code)]
pub fn is_hevc_available() -> bool {
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    {
        return media_foundation::probe_hevc();
    }
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    {
        return true;
    }
    false
}

/// Check whether the host can hardware-decode H.264 via a platform API
/// (VideoToolbox, Media Foundation, etc. — excludes VA-API which is incomplete).
#[allow(unreachable_code)]
pub fn is_h264_hw_available() -> bool {
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    {
        return true;
    }
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    {
        return true;
    }
    false
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
            let g_lo = vshrq_n_s32(
                vaddq_s32(
                    vsubq_s32(
                        vsubq_s32(c_lo, vmull_s16(c208, cr_lo)),
                        vmull_s16(c100, cb_lo),
                    ),
                    half,
                ),
                8,
            );
            let b_lo = vshrq_n_s32(vaddq_s32(vaddq_s32(c_lo, vmull_s16(c516, cb_lo)), half), 8);

            let y_hi = vget_high_s16(y_adj);
            let cb_hi = vget_high_s16(cb_adj);
            let cr_hi = vget_high_s16(cr_adj);
            let c_hi = vmull_s16(c298, y_hi);
            let r_hi = vshrq_n_s32(vaddq_s32(vaddq_s32(c_hi, vmull_s16(c409, cr_hi)), half), 8);
            let g_hi = vshrq_n_s32(
                vaddq_s32(
                    vsubq_s32(
                        vsubq_s32(c_hi, vmull_s16(c208, cr_hi)),
                        vmull_s16(c100, cb_hi),
                    ),
                    half,
                ),
                8,
            );
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
#[target_feature(enable = "sse4.1")]
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

    let c298 = _mm_set1_epi32(298);
    let c409 = _mm_set1_epi32(409);
    let c100 = _mm_set1_epi32(100);
    let c208 = _mm_set1_epi32(208);
    let c516 = _mm_set1_epi32(516);
    let half32 = _mm_set1_epi32(128);
    let zero = _mm_setzero_si128();

    for row in 0..h {
        let y_row = y_ptr.add(row * y_stride);
        let uv_row = uv_ptr.add((row / 2) * uv_stride);
        let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
        let mut col = 0usize;

        while col + 4 <= w {
            let y4_bytes: [u8; 4] = [
                *y_row.add(col),
                *y_row.add(col + 1),
                *y_row.add(col + 2),
                *y_row.add(col + 3),
            ];
            let y4 = _mm_set_epi32(
                y4_bytes[3] as i32 - 16,
                y4_bytes[2] as i32 - 16,
                y4_bytes[1] as i32 - 16,
                y4_bytes[0] as i32 - 16,
            );

            let cb0 = *uv_row.add((col / 2) * 2) as i32 - 128;
            let cr0 = *uv_row.add((col / 2) * 2 + 1) as i32 - 128;
            let cb1 = *uv_row.add(((col + 2) / 2) * 2) as i32 - 128;
            let cr1 = *uv_row.add(((col + 2) / 2) * 2 + 1) as i32 - 128;
            let cb = _mm_set_epi32(cb1, cb1, cb0, cb0);
            let cr = _mm_set_epi32(cr1, cr1, cr0, cr0);

            let c_val = _mm_mullo_epi32(c298, y4);
            let r32 = _mm_srai_epi32::<8>(_mm_add_epi32(
                _mm_add_epi32(c_val, _mm_mullo_epi32(c409, cr)),
                half32,
            ));
            let g32 = _mm_srai_epi32::<8>(_mm_add_epi32(
                _mm_sub_epi32(
                    _mm_sub_epi32(c_val, _mm_mullo_epi32(c208, cr)),
                    _mm_mullo_epi32(c100, cb),
                ),
                half32,
            ));
            let b32 = _mm_srai_epi32::<8>(_mm_add_epi32(
                _mm_add_epi32(c_val, _mm_mullo_epi32(c516, cb)),
                half32,
            ));

            let r16 = _mm_packs_epi32(r32, zero);
            let g16 = _mm_packs_epi32(g32, zero);
            let b16 = _mm_packs_epi32(b32, zero);
            let r_u8 = _mm_packus_epi16(_mm_max_epi16(r16, zero), zero);
            let g_u8 = _mm_packus_epi16(_mm_max_epi16(g16, zero), zero);
            let b_u8 = _mm_packus_epi16(_mm_max_epi16(b16, zero), zero);

            let mut r_arr = [0u8; 4];
            let mut g_arr = [0u8; 4];
            let mut b_arr = [0u8; 4];
            std::ptr::copy_nonoverlapping(
                &r_u8 as *const __m128i as *const u8,
                r_arr.as_mut_ptr(),
                4,
            );
            std::ptr::copy_nonoverlapping(
                &g_u8 as *const __m128i as *const u8,
                g_arr.as_mut_ptr(),
                4,
            );
            std::ptr::copy_nonoverlapping(
                &b_u8 as *const __m128i as *const u8,
                b_arr.as_mut_ptr(),
                4,
            );
            for i in 0..4 {
                let dst = (col + i) * 3;
                dst_row[dst] = r_arr[i];
                dst_row[dst + 1] = g_arr[i];
                dst_row[dst + 2] = b_arr[i];
            }

            col += 4;
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
