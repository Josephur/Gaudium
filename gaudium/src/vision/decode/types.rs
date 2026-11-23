//! Shared types of the host image decoders.

/// Limits checked from an image's header, before any pixel buffer is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Largest canvas (width x height) that is decoded.
    pub max_pixels: u64,
}

impl DecodeLimits {
    /// Refuses a canvas outside the limits (the verified canvas rule). Every decoder calls this
    /// as soon as the header gives the size, before it allocates anything that grows with the
    /// image.
    pub fn check_canvas(&self, width: usize, height: usize) -> Result<(), String> {
        crate::core::vision::check_canvas(width, height, self.max_pixels)
    }
}

/// A decoded still image: RGB8, rows top to bottom, 3 bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Still {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    /// The file's EXIF block (TIFF structure, starting at the byte-order mark), if it has one.
    pub exif: Option<Vec<u8>>,
}

/// What a cheap pass over a file tells without decoding pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    pub width: usize,
    pub height: usize,
    /// Display time of every frame in milliseconds, in file order (one entry for a still image).
    pub delays_ms: Vec<u32>,
    pub exif: Option<Vec<u8>>,
}
