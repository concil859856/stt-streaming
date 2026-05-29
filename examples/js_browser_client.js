// USAGE EXAMPLE: minimal browser client for stt-streaming.
//
// Pure JS, no library. Captures the user's microphone via getUserMedia +
// AudioWorklet, downmixes to mono, downsamples from the browser's native
// rate (typically 48 kHz) to 16 kHz, converts to PCM16 little-endian, and
// streams the raw bytes as WebSocket binary frames.
//
// Drop this in a page that also serves the worklet snippet below as
// `pcm-worklet.js` (the snippet at the bottom of this file).
//
// Browsers do not allow setting custom request headers on WS upgrades,
// so the API key is passed as a query string (`?api_key=...`). The server
// MUST also accept the key this way; see server.py.

const WS_URL = "ws://localhost:8114/v1/stream";
const API_KEY = "test_key_local_only";
const TARGET_SR = 16000;

async function startStreaming() {
  // 1. Open the WS first so the server is ready before we capture audio.
  const ws = new WebSocket(`${WS_URL}?api_key=${encodeURIComponent(API_KEY)}`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    // 2. Send the mandatory `start` message describing the stream format.
    ws.send(JSON.stringify({
      type: "start",
      language: "auto",
      sample_rate: TARGET_SR,
      encoding: "pcm_s16le",
      enable_partials: true,
    }));
  });

  ws.addEventListener("message", (ev) => {
    if (typeof ev.data !== "string") return;
    const msg = JSON.parse(ev.data);
    if (msg.type === "ready") console.log("ready:", msg);
    else if (msg.type === "partial") console.log("...", msg.text);
    else if (msg.type === "final") console.log(">>>", msg.text);
    else if (msg.type === "error") console.error("server error:", msg);
  });

  // 3. Capture mic. We let the browser pick its native sample rate and
  //    downsample inside the worklet so getUserMedia hits the fast path.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-worklet", {
    processorOptions: { targetSampleRate: TARGET_SR, sourceSampleRate: ctx.sampleRate },
  });

  // 4. The worklet posts PCM16 LE ArrayBuffers to the main thread; we
  //    forward each one as a WS binary frame.
  node.port.onmessage = (ev) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(ev.data);
  };

  src.connect(node);
  // We don't want to play the mic back, so don't connect node to destination.

  // 5. Expose a stop() that flushes a `close` and tears down the graph.
  window._asrStop = () => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "close" }));
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
  };
}

// To run: call startStreaming() from a user-gesture handler (e.g. a button).
// document.querySelector("#start").addEventListener("click", startStreaming);

/* ---------------- pcm-worklet.js (serve as a sibling file) ----------------

class PCMWorklet extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const o = opts.processorOptions || {};
    this.targetSR = o.targetSampleRate || 16000;
    this.sourceSR = o.sourceSampleRate || sampleRate;
    this.ratio = this.sourceSR / this.targetSR;
    this.acc = 0;
    this.buf = [];
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // Linear-interp downsample to targetSR, mono.
    for (let i = 0; i < ch.length; i += this.ratio) {
      const idx = Math.floor(i);
      const sample = ch[idx] || 0;
      this.buf.push(Math.max(-1, Math.min(1, sample)));
    }
    // Flush ~20 ms (320 samples @ 16 kHz) at a time.
    while (this.buf.length >= 320) {
      const out = new Int16Array(320);
      for (let i = 0; i < 320; i++) out[i] = this.buf[i] * 0x7fff;
      this.port.postMessage(out.buffer, [out.buffer]);
      this.buf = this.buf.slice(320);
    }
    return true;
  }
}
registerProcessor("pcm-worklet", PCMWorklet);

----------------------------------------------------------------------- */
