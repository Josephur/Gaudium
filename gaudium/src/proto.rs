//! Binary Unix-socket wire protocol between the HTTP server and the Gaudi 2 engine worker.
//!
//! Wire-compatible with `serve_proto.hpp`: every frame is little-endian
//! `[u32 length (1 + payload.len())][u8 msg_type][payload bytes]`, bounded by `MAX_FRAME = 1 GiB`.

use std::io::{self, Read, Write};

use crate::core::budget::advertised_served_ctx;

/// Maximum allowed frame length (1 GiB: multi-image requests are < 200 MiB of f32 pixels).
pub const MAX_FRAME: u32 = 1 << 30;

/// Maximum images permitted in one `Submit` frame.
pub const MAX_IMAGES_PER_REQUEST: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    Submit = 1,
    Cancel = 2,
    Shutdown = 3,
    Ping = 4,
    State = 10,
    Event = 11,
    Pong = 12,
}

impl TryFrom<u8> for MsgType {
    type Error = ProtoError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Self::Submit),
            2 => Ok(Self::Cancel),
            3 => Ok(Self::Shutdown),
            4 => Ok(Self::Ping),
            10 => Ok(Self::State),
            11 => Ok(Self::Event),
            12 => Ok(Self::Pong),
            other => Err(ProtoError::UnknownMsgType(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EvKind {
    Started = 0,
    Progress = 1,
    Tokens = 2,
    Done = 3,
    Error = 4,
}

impl TryFrom<u8> for EvKind {
    type Error = ProtoError;

    fn try_from(v: u8) -> Result<Self, ProtoError> {
        match v {
            0 => Ok(Self::Started),
            1 => Ok(Self::Progress),
            2 => Ok(Self::Tokens),
            3 => Ok(Self::Done),
            4 => Ok(Self::Error),
            other => Err(ProtoError::UnknownEvKind(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Finish {
    Stop = 0,
    Length = 1,
    Cancelled = 2,
    Error = 3,
}

impl TryFrom<u8> for Finish {
    type Error = ProtoError;

    fn try_from(v: u8) -> Result<Self, ProtoError> {
        match v {
            0 => Ok(Self::Stop),
            1 => Ok(Self::Length),
            2 => Ok(Self::Cancelled),
            3 => Ok(Self::Error),
            other => Err(ProtoError::UnknownFinish(other)),
        }
    }
}

#[derive(Debug)]
pub enum ProtoError {
    Truncated,
    OversizedFrame(u32),
    TooManyImages(u32),
    InvalidUtf8,
    UnknownMsgType(u8),
    UnknownEvKind(u8),
    UnknownFinish(u8),
    Io(io::Error),
}

impl From<io::Error> for ProtoError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Computes advertised served context (`lmax - 128`) via the verified Bend 2 budget core.
pub fn served_ctx(lmax: u64) -> u64 {
    advertised_served_ctx(lmax)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImagePayload {
    pub t: i32,
    pub h: i32,
    pub w: i32,
    pub span0: i32,
    pub ntok: i32,
    pub pixels: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Submit {
    pub id: u64,
    pub method: String,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub coupled: bool,
    pub draft_trunc: bool,
    pub max_new: i32,
    pub stop_ids: Vec<i32>,
    pub prompt: Vec<i32>,
    pub images: Vec<ImagePayload>,
    /// v2 trailing fields (absent from v1 clients: defaults to cache on, no snapshot hints).
    pub cache_prompt: bool,
    pub snapshot_at: Vec<i32>,
}

impl Default for Submit {
    fn default() -> Self {
        Self {
            id: 0,
            method: String::new(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            coupled: true,
            draft_trunc: true,
            max_new: -1,
            stop_ids: Vec::new(),
            prompt: Vec::new(),
            images: Vec::new(),
            cache_prompt: true,
            snapshot_at: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    pub cache_n: i32,
    pub prompt_n: i32,
    pub predicted_n: i32,
    pub launches: i32,
    pub draft_n: i32,
    pub draft_accepted: i32,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub id: u64,
    pub kind: EvKind,
    pub tokens: Vec<i32>,
    pub processed: i32,
    pub total: i32,
    pub elapsed_ms: f64,
    pub timings: Timings,
    pub finish: Finish,
    pub message: String,
    pub extra: String,
}

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new(msg_type: MsgType) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(msg_type as u8);
        Self { buf }
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn vec_i32(&mut self, v: &[i32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.i32(x);
        }
    }

    fn vec_f32(&mut self, v: &[f32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.f32(x);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let len = (self.buf.len() - 4) as u32;
        self.buf[0..4].copy_from_slice(&len.to_le_bytes());
        self.buf
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ProtoError> {
        let end = self.pos.checked_add(n).ok_or(ProtoError::Truncated)?;
        if end > self.b.len() {
            return Err(ProtoError::Truncated);
        }
        let slice = &self.b[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ProtoError> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> Result<i32, ProtoError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ProtoError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ProtoError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ProtoError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, ProtoError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn str(&mut self) -> Result<String, ProtoError> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| ProtoError::InvalidUtf8)
    }

    fn vec_i32(&mut self) -> Result<Vec<i32>, ProtoError> {
        let n = self.u32()? as usize;
        let byte_len = n.checked_mul(4).ok_or(ProtoError::Truncated)?;
        let bytes = self.take(byte_len)?;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }

    fn vec_f32(&mut self) -> Result<Vec<f32>, ProtoError> {
        let n = self.u32()? as usize;
        let byte_len = n.checked_mul(4).ok_or(ProtoError::Truncated)?;
        let bytes = self.take(byte_len)?;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}

pub fn encode_submit(s: &Submit) -> Vec<u8> {
    let mut w = Writer::new(MsgType::Submit);
    w.u64(s.id);
    w.str(&s.method);
    w.f32(s.temperature);
    w.i32(s.top_k);
    w.f32(s.top_p);
    w.f32(s.min_p);
    w.u64(s.seed);
    w.u8(u8::from(s.coupled));
    w.u8(u8::from(s.draft_trunc));
    w.i32(s.max_new);
    w.vec_i32(&s.stop_ids);
    w.vec_i32(&s.prompt);
    w.u32(s.images.len() as u32);
    for im in &s.images {
        w.i32(im.t);
        w.i32(im.h);
        w.i32(im.w);
        w.i32(im.span0);
        w.i32(im.ntok);
        w.vec_f32(&im.pixels);
    }
    w.u8(u8::from(s.cache_prompt));
    w.vec_i32(&s.snapshot_at);
    w.finish()
}

pub fn decode_submit(payload: &[u8]) -> Result<Submit, ProtoError> {
    let mut r = Reader::new(payload);
    let id = r.u64()?;
    let method = r.str()?;
    let temperature = r.f32()?;
    let top_k = r.i32()?;
    let top_p = r.f32()?;
    let min_p = r.f32()?;
    let seed = r.u64()?;
    let coupled = r.u8()? != 0;
    let draft_trunc = r.u8()? != 0;
    let max_new = r.i32()?;
    let stop_ids = r.vec_i32()?;
    let prompt = r.vec_i32()?;
    let ni = r.u32()?;
    if ni > MAX_IMAGES_PER_REQUEST {
        return Err(ProtoError::TooManyImages(ni));
    }
    let mut images = Vec::with_capacity(ni as usize);
    for _ in 0..ni {
        images.push(ImagePayload {
            t: r.i32()?,
            h: r.i32()?,
            w: r.i32()?,
            span0: r.i32()?,
            ntok: r.i32()?,
            pixels: r.vec_f32()?,
        });
    }
    let (cache_prompt, snapshot_at) = if r.remaining() > 0 {
        (r.u8()? != 0, r.vec_i32()?)
    } else {
        (true, Vec::new())
    };
    Ok(Submit {
        id,
        method,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        coupled,
        draft_trunc,
        max_new,
        stop_ids,
        prompt,
        images,
        cache_prompt,
        snapshot_at,
    })
}

pub fn encode_event(e: &Event) -> Vec<u8> {
    let mut w = Writer::new(MsgType::Event);
    w.u64(e.id);
    w.u8(e.kind as u8);
    w.vec_i32(&e.tokens);
    w.i32(e.processed);
    w.i32(e.total);
    w.f64(e.elapsed_ms);
    w.i32(e.timings.cache_n);
    w.i32(e.timings.prompt_n);
    w.f64(e.timings.prompt_ms);
    w.i32(e.timings.predicted_n);
    w.f64(e.timings.predicted_ms);
    w.i32(e.timings.launches);
    w.i32(e.timings.draft_n);
    w.i32(e.timings.draft_accepted);
    w.u8(e.finish as u8);
    w.str(&e.message);
    w.str(&e.extra);
    w.finish()
}

pub fn decode_event(payload: &[u8]) -> Result<Event, ProtoError> {
    let mut r = Reader::new(payload);
    let id = r.u64()?;
    let kind = EvKind::try_from(r.u8()?)?;
    let tokens = r.vec_i32()?;
    let processed = r.i32()?;
    let total = r.i32()?;
    let elapsed_ms = r.f64()?;
    let cache_n = r.i32()?;
    let prompt_n = r.i32()?;
    let prompt_ms = r.f64()?;
    let predicted_n = r.i32()?;
    let predicted_ms = r.f64()?;
    let launches = r.i32()?;
    let draft_n = r.i32()?;
    let draft_accepted = r.i32()?;
    let finish = Finish::try_from(r.u8()?)?;
    let message = r.str()?;
    let extra = r.str()?;
    Ok(Event {
        id,
        kind,
        tokens,
        processed,
        total,
        elapsed_ms,
        timings: Timings {
            cache_n,
            prompt_n,
            predicted_n,
            launches,
            draft_n,
            draft_accepted,
            prompt_ms,
            predicted_ms,
        },
        finish,
        message,
        extra,
    })
}

pub fn encode_u64(msg_type: MsgType, val: u64) -> Vec<u8> {
    let mut w = Writer::new(msg_type);
    w.u64(val);
    w.finish()
}

pub fn decode_u64(payload: &[u8]) -> Result<u64, ProtoError> {
    Reader::new(payload).u64()
}

pub fn encode_text(msg_type: MsgType, text: &str) -> Vec<u8> {
    let mut w = Writer::new(msg_type);
    w.str(text);
    w.finish()
}

pub fn decode_text(payload: &[u8]) -> Result<String, ProtoError> {
    Reader::new(payload).str()
}

pub fn send_frame<W: Write>(mut stream: W, frame: &[u8]) -> Result<(), ProtoError> {
    stream.write_all(frame)?;
    Ok(())
}

pub fn recv_frame<R: Read>(mut stream: R) -> Result<(MsgType, Vec<u8>), ProtoError> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr)?;
    let len = u32::from_le_bytes(hdr);
    if len < 1 || len > MAX_FRAME {
        return Err(ProtoError::OversizedFrame(len));
    }
    let mut tbuf = [0u8; 1];
    stream.read_exact(&mut tbuf)?;
    let msg_type = MsgType::try_from(tbuf[0])?;
    let mut payload = vec![0u8; (len - 1) as usize];
    if !payload.is_empty() {
        stream.read_exact(&mut payload)?;
    }
    Ok((msg_type, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrips_submit_v2_and_v1_and_event_frames() {
        let sub = Submit {
            id: 42,
            method: "adaptive".to_owned(),
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: 0xDEAD_BEEF_CAFE_BABE,
            coupled: true,
            draft_trunc: false,
            max_new: 512,
            stop_ids: vec![151645, 151643],
            prompt: vec![1, 2, 3, 4],
            images: vec![ImagePayload {
                t: 1,
                h: 28,
                w: 28,
                span0: 2,
                ntok: 196,
                pixels: vec![0.25, -0.5, 1.0],
            }],
            cache_prompt: false,
            snapshot_at: vec![512, 1024],
        };
        let frame = encode_submit(&sub);
        let mut cursor = Cursor::new(frame);
        let (mt, payload) = recv_frame(&mut cursor).unwrap();
        assert_eq!(mt, MsgType::Submit);
        assert_eq!(decode_submit(&payload).unwrap(), sub);

        // v1 backward compatibility: strip trailing v2 bytes (1 byte cache_prompt + 4 byte len + 8 bytes)
        let v1_payload = &payload[..payload.len() - 13];
        let dec_v1 = decode_submit(v1_payload).unwrap();
        assert!(dec_v1.cache_prompt);
        assert!(dec_v1.snapshot_at.is_empty());

        let ev = Event {
            id: 42,
            kind: EvKind::Done,
            tokens: vec![99, 100],
            processed: 4,
            total: 4,
            elapsed_ms: 12.5,
            timings: Timings {
                cache_n: 512,
                prompt_n: 4,
                predicted_n: 2,
                launches: 1,
                draft_n: 4,
                draft_accepted: 1,
                prompt_ms: 3.2,
                predicted_ms: 9.3,
            },
            finish: Finish::Stop,
            message: String::new(),
            extra: r#"{"method":"adaptive"}"#.to_owned(),
        };
        let ev_frame = encode_event(&ev);
        let (emt, epayload) = recv_frame(Cursor::new(ev_frame)).unwrap();
        assert_eq!(emt, MsgType::Event);
        assert_eq!(decode_event(&epayload).unwrap(), ev);

        // Cancel & State text frames
        let cancel_frame = encode_u64(MsgType::Cancel, 77);
        let (cmt, cp) = recv_frame(Cursor::new(cancel_frame)).unwrap();
        assert_eq!(cmt, MsgType::Cancel);
        assert_eq!(decode_u64(&cp).unwrap(), 77);

        let st_frame = encode_text(MsgType::State, r#"{"state":"ready"}"#);
        let (smt, sp) = recv_frame(Cursor::new(st_frame)).unwrap();
        assert_eq!(smt, MsgType::State);
        assert_eq!(decode_text(&sp).unwrap(), r#"{"state":"ready"}"#);
        assert_eq!(served_ctx(262_144), 262_016);
    }

    #[test]
    fn rejects_oversized_and_truncated_frames() {
        let mut bad_hdr = Vec::new();
        bad_hdr.extend_from_slice(&(MAX_FRAME + 1).to_le_bytes());
        assert!(matches!(
            recv_frame(Cursor::new(bad_hdr)),
            Err(ProtoError::OversizedFrame(_))
        ));
        assert!(matches!(decode_submit(&[0, 1, 2]), Err(ProtoError::Truncated)));
    }
}
