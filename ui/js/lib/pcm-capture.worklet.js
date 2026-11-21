// The microphone's samples, as they arrive (an AudioWorklet: runs on the audio thread). The channels of each render quantum are
// averaged to mono and posted to the page in blocks of 2048 frames; "flush" posts what is left and then "done".
// No codec in between: the page writes the WAV itself (lib/audio.js), so what is sent is what the microphone gave.
class GaudiumPcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(2048); this.n = 0; this.live = true;
    this.port.onmessage = (e) => {
      if (e.data !== 'flush') return;
      if (this.n) this.port.postMessage(this.buf.slice(0, this.n));
      this.n = 0; this.live = false; this.port.postMessage('done');
    };
  }
  process(inputs) {
    const ch = inputs[0];
    if (!this.live) return false;
    if (ch && ch.length) {
      const k = ch.length, len = ch[0].length;
      for (let i = 0; i < len; i++) {
        let s = 0; for (let c = 0; c < k; c++) s += ch[c][i];
        this.buf[this.n++] = s / k;
        if (this.n === this.buf.length) { this.port.postMessage(this.buf, [this.buf.buffer]); this.buf = new Float32Array(2048); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor('gaudium-pcm-capture', GaudiumPcmCapture);
