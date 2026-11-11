// SpaceAudio v0.2 — Depth Anything V2 Small + SegFormer-B0(ADE20K) 실제 추론. 접근물체/신호등은 미구현.
import { pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
const $ = id => document.getElementById(id);
const v = $('v'), st = $('status'), go = $('go'), octx = $('o').getContext('2d');
const W = 256, H = 192, C = 12, R = 12;
const HFOV = 60 * Math.PI / 180, SPREAD = 1.6;   // 가정값: 카메라 수평 시야 60°, 음향 무대 1.6배 확장
const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
const cx = cv.getContext('2d', { willReadFrequently: true });
let ac, noise, depther, seg, stream, running = false, facing = 'environment', beta = null, fileMode = false;
let frame = 0, segBusy = false, segFloor = null, fitA = 0.05, fitB = 0, conf = 0, voices = [], beacon = null;
const colS = new Float32Array(C), urg = new Uint8Array(C), cellSev = new Float32Array(C * R);
const nextAt = {}, speakAt = {}, logs = [], quiz = { n: 0, ok: 0, cur: null };
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const colAz = c => ((c + 0.5) / C - 0.5) * HFOV * SPREAD;               // 라디안, +우측
const clock = az => { let h = Math.round(az * 180 / Math.PI / 30); return ((h + 12) % 12 || 12) + '시 방향'; };
const h2az = h => (h > 6 ? h - 12 : h) * 30 * Math.PI / 180;
const FREQ = [900, 550, 330];                                           // 하부/중부/상부 음색

function say(t, k, cd) {
  const n = performance.now(); if (speakAt[k] && n - speakAt[k] < cd) return; speakAt[k] = n;
  speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(t); u.lang = 'ko-KR'; speechSynthesis.speak(u);
}
// ---------- 3D 공간 음향: 연속 방위각(HRTF) + 높이(y) ----------
function initAudio() {
  ac = new (window.AudioContext || window.webkitAudioContext)();
  noise = ac.createBuffer(1, ac.sampleRate * 0.15, ac.sampleRate);
  const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  setInterval(tick, 40);
}
function spatialBeep(az, el, freq, gain, kind = 'noise') {   // 광대역 노이즈 버스트가 순음보다 방향 정위가 좋음
  const t = ac.currentTime, g = ac.createGain();
  const p = new PannerNode(ac, { panningModel: 'HRTF', distanceModel: 'inverse', refDistance: 1,
    positionX: 1.5 * Math.sin(az), positionY: el, positionZ: -1.5 * Math.cos(az) });
  let s;
  if (kind === 'noise') { s = ac.createBufferSource(); s.buffer = noise; const f = ac.createBiquadFilter();
    f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = 1.5; s.connect(f); f.connect(g); gain *= 4; }
  else { s = ac.createOscillator(); s.type = kind; s.frequency.value = freq; s.connect(g); }
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(clamp(gain), t + 0.008);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
  g.connect(p); p.connect(ac.destination); s.start(t); s.stop(t + 0.15);
}
function tick() {
  if (!running) return; const n = performance.now();
  for (const o of voices) {           // 위험도↑ → 반복 간격↓ (음량으로 거리 표현 X)
    if (n < (nextAt[o.id] || 0)) continue;
    spatialBeep(o.az, [-0.6, 0, 0.6][o.band], FREQ[o.band], 0.15 + 0.25 * o.sev, o.sev > 0.8 ? 'square' : 'noise');
    nextAt[o.id] = n + 1000 - 850 * o.sev;
  }
  if (beacon != null && n > (nextAt.bc || 0)) { spatialBeep(beacon, 0, 660, 0.1, 'sine'); nextAt.bc = n + 1400; }
}
// ---------- 분석 ----------
function fit(pts) {
  const total = pts.length; let a = 0, b = 0;
  for (let it = 0; it < 3 && pts.length >= 8; it++) {
    const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    let sxy = 0, sxx = 0; pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
    a = sxy / (sxx + 1e-9); b = my - a * mx;
    const res = pts.map(p => p[1] - (a * p[0] + b)), sd = Math.sqrt(res.reduce((s, x) => s + x * x, 0) / n) + 1e-6;
    pts = pts.filter((p, i) => Math.abs(res[i]) < 1.5 * sd);
  }
  return { a, b, inlier: total ? pts.length / total : 0 };
}
function analyze(d) {
  const w = d.width, h = d.height, g = new Float32Array(C * R);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    let s = 0, n = 0;
    for (let y = (r * h / R) | 0; y < ((r + 1) * h / R | 0); y++) for (let x = (c * w / C) | 0; x < ((c + 1) * w / C | 0); x++) { s += d.data[y * w + x]; n++; }
    g[r * C + c] = s / n / 255;
  }
  const so = Float32Array.from(g).sort(), lo = so[(C * R * 0.05) | 0], hi = so[(C * R * 0.95) | 0];
  for (let i = 0; i < g.length; i++) g[i] = (g[i] - lo) / (hi - lo + 1e-6);
  // 바닥 모델: SegFormer가 '바닥'이라 한 셀로 적합(없으면 하단 중앙 폴백)
  let pts = [], src = 'depth';
  if (segFloor) { for (let r = 5; r < R; r++) for (let c = 0; c < C; c++) if (segFloor[r * C + c] > 0.6) pts.push([r, g[r * C + c]]); if (pts.length >= 10) src = 'seg+depth'; }
  if (src === 'depth') { pts = []; for (let r = 7; r < R; r++) for (let c = 3; c < 9; c++) pts.push([r, g[r * C + c]]); }
  const f = fit(pts), ok = f.a > 0.01;
  if (ok) { fitA = 0.7 * fitA + 0.3 * f.a; fitB = 0.7 * fitB + 0.3 * f.b; }        // 시간적 평활
  const pitchOK = beta == null || Math.abs(beta - 90) < 35;
  conf = (ok ? f.inlier : 0) * (pitchOK ? 1 : 0.5) * (src === 'seg+depth' ? 1 : 0.8);
  const dRef = fitA * 9 + fitB; let fl = 0, conflict = 0;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const i = r * C + c, val = g[i];
    let s = r >= 8 ? clamp((val - (fitA * r + fitB) - 0.12) / 0.25) : clamp((val - dRef - 0.1) / 0.25);
    if (segFloor && r >= 8) {
      const sf = segFloor[i];
      if (sf > 0.7) { fl++; if (s > 0.6) conflict++; s *= 0.6; }                  // 센서 불일치는 불확실성↑
      else if (sf < 0.2 && s > 0.2) s = clamp(s * 1.3);                          // 비바닥 + 깊이 근접 → 강화
    }
    cellSev[i] = s;
  }
  if (fl > 8) conf *= 1 - 0.8 * (conflict / fl);
  return src;
}
function update() {
  const raw = new Float32Array(C), rb = new Uint8Array(C);
  for (let c = 0; c < C; c++) {
    let best = -1;
    for (let b = 0; b < 3; b++) {   // 0 하부, 1 중부, 2 상부
      const rows = b === 0 ? [8, 12] : b === 1 ? [4, 8] : [0, 4], v2 = [];
      for (let r = rows[0]; r < rows[1]; r++) v2.push(cellSev[r * C + c]);
      v2.sort((x, y) => y - x); const m = (v2[0] + v2[1]) / 2;
      if (m > best) { best = m; rb[c] = b; }
    }
    raw[c] = best;
    colS[c] = raw[c] > colS[c] ? 0.7 * raw[c] + 0.3 * colS[c] : 0.4 * raw[c] + 0.6 * colS[c]; // 빠른 상승
    urg[c] = colS[c] > 0.85 ? urg[c] + 1 : 0;
  }
  const peaks = [];   // 지역 최대 열에서 최대 3개 음원
  for (let c = 0; c < C; c++) if (colS[c] > 0.3 && colS[c] >= (colS[c - 1] ?? 0) && colS[c] >= (colS[c + 1] ?? 0)) peaks.push(c);
  peaks.sort((a, b) => colS[b] - colS[a]);
  const chosen = []; for (const c of peaks) if (chosen.every(k => Math.abs(k - c) >= 2) && chosen.length < 3) chosen.push(c);
  voices = chosen.map(c => ({ id: c, az: colAz(c), sev: colS[c], band: rb[c] }));
  const u = chosen.find(c => urg[c] >= 2); if (u != null) say(clock(colAz(u)) + ' 장애물, 멈추세요', 'u', 3000);
  // 비콘: 중앙 경로가 막혔을 때 가장 열린 방향을 부드러운 사인음으로 안내
  beacon = null;
  if (Math.max(...colS.slice(4, 8)) > 0.3) {
    let bc = -1, bs = 9;
    for (let c = 1; c < C - 1; c++) { const s = (colS[c - 1] + colS[c] + colS[c + 1]) / 3 + Math.abs(c - 5.5) * 0.01; if (s < bs) { bs = s; bc = c; } }
    if (bs < 0.25) beacon = colAz(bc);
  }
  if (conf < 0.45) say('판단이 불확실합니다. 카메라를 천천히 움직여 주세요', 'unc', 6000);
}
function draw() {
  octx.clearRect(0, 0, W, H); const cw = W / C, ch = H / R;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const s = cellSev[r * C + c];
    octx.fillStyle = conf < 0.45 ? 'rgba(255,200,0,.25)' : `rgba(${255 * s | 0},${255 * (1 - s) | 0},60,.35)`;
    octx.fillRect(c * cw, r * ch, cw - 1, ch - 1);
  }
}
// ---------- 모델 / 카메라 / 루프 ----------
const prog = n => p => { if (p.status === 'progress') st.textContent = `${n} 다운로드 ${p.progress | 0}%`; };
async function loadModels() {
  const id = 'onnx-community/depth-anything-v2-small';
  if (navigator.gpu) { try { depther = await pipeline('depth-estimation', id, { device: 'webgpu', progress_callback: prog('깊이모델') }); } catch (e) { console.warn(e); } }
  depther ??= await pipeline('depth-estimation', id, { device: 'wasm', dtype: 'q8', progress_callback: prog('깊이모델') });
  try { seg = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-ade-512-512', { device: 'wasm', dtype: 'q8', progress_callback: prog('분할모델') }); }
  catch (e) { console.warn('SegFormer 로드 실패 → 깊이 단독 모드', e); seg = null; }
}
function runSeg() {
  segBusy = true;
  seg(RawImage.fromCanvas(cv)).then(res => {
    const f = new Float32Array(C * R);
    for (const m of res) { if (!/^(floor|road|sidewalk|rug|path)/.test(m.label)) continue; const k = m.mask;
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) for (let j = 0; j < 9; j++) {
        const x = ((c + (j % 3 + 0.5) / 3) * k.width / C) | 0, y = ((r + (((j / 3) | 0) + 0.5) / 3) * k.height / R) | 0;
        if (k.data[y * k.width + x] > 127) f[r * C + c] += 1 / 9;
      } }
    segFloor = f;
  }).catch(console.warn).finally(() => { segBusy = false; });
}
async function startCam() {
  stream?.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 640 } }, audio: false });
  v.srcObject = stream; await v.play();
}
async function loop() {
  let last = performance.now();
  while (running) {
    const t0 = performance.now(); frame++;
    cx.drawImage(v, 0, 0, W, H);
    if (seg && !segBusy && frame % 4 === 1) runSeg();
    const out = await depther(RawImage.fromCanvas(cv));
    const src = analyze(out.depth); update(); draw();
    const dt = performance.now() - t0, fps = 1000 / (performance.now() - last); last = performance.now();
    logs.push([Date.now(), dt | 0, fps.toFixed(1), conf.toFixed(2), src, ...Array.from(colS, x => x.toFixed(2))]);
    st.textContent = `추론 ${dt | 0}ms · ${fps.toFixed(1)}fps · 신뢰도 ${conf * 100 | 0}% · ${src}${seg ? '' : '(분할 없음)'}${beacon != null ? ' · 열린 길 ' + clock(beacon) : ''}`;
    await new Promise(r => setTimeout(r, 0));
  }
}
function stop() { running = false; stream?.getTracks().forEach(t => t.stop()); voices = []; beacon = null; v.pause();
  go.textContent = '보조 시작'; go.classList.remove('on'); st.textContent = '정지됨'; say('보조 기능을 정지합니다', 's', 0); }
go.onclick = async () => {
  if (running) return stop();
  try {
    if (!ac) initAudio(); await ac.resume();
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') await DeviceOrientationEvent.requestPermission();
    addEventListener('deviceorientation', e => { beta = e.beta; });
    if (!depther) { st.textContent = '모델을 불러오는 중…'; say('모델을 불러옵니다', 'l', 0); await loadModels(); }
    fileMode ? await v.play() : await startCam();
    running = true; go.textContent = '보조 정지'; go.classList.add('on'); say('보조 기능을 시작합니다', 's', 0); loop();
  } catch (e) { st.textContent = '오류: ' + e.message; console.error(e); say('시작하지 못했습니다', 'e', 0); }
};
$('flip').onclick = async () => { if (fileMode) return; facing = facing === 'environment' ? 'user' : 'environment'; if (running) await startCam(); };
$('file').onchange = e => { const f = e.target.files[0]; if (!f) return; fileMode = true; stream?.getTracks().forEach(t => t.stop());
  v.srcObject = null; v.src = URL.createObjectURL(f); v.loop = true; v.muted = true; st.textContent = '영상 파일 선택됨. "보조 시작"을 누르세요'; };
$('log').onclick = () => { const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['t,ms,fps,conf,src,' + Array.from({ length: C }, (_, i) => 'c' + i) + '\n' + logs.map(r => r.join(',')).join('\n') + '\n\nquiz_ok,quiz_n\n' + quiz.ok + ',' + quiz.n], { type: 'text/csv' })));
  a.download = 'spaceaudio-log.csv'; a.click(); };
// ---------- 3D 소리 테스트 & 방향 인식 퀴즈(방향 인식률 측정) ----------
$('test').onclick = async () => {
  if (!ac) initAudio(); await ac.resume();
  [3, 9, 2, 10, 12, 1, 11].forEach((h, i) => setTimeout(() => {
    spatialBeep(h2az(h), 0, 900, 0.5); setTimeout(() => spatialBeep(h2az(h), 0, 900, 0.5), 250);
    st.textContent = h + '시 방향'; setTimeout(() => { const u = new SpeechSynthesisUtterance(h + '시'); u.lang = 'ko-KR'; speechSynthesis.speak(u); }, 600);
  }, i * 1800));
};
const qd = $('quiz');
[9, 10, 11, 12, 1, 2, 3].forEach(h => { const b = document.createElement('button'); b.textContent = h + '시';
  b.onclick = () => { if (quiz.cur == null) return; quiz.n++; if (h === quiz.cur) quiz.ok++;
    st.textContent = `${h === quiz.cur ? '정답' : '오답, 정답 ' + quiz.cur + '시'} · 방향 인식률 ${quiz.ok}/${quiz.n} (${(100 * quiz.ok / quiz.n) | 0}%)`; next(); };
  qd.appendChild(b); });
function next() { quiz.cur = [9, 10, 11, 12, 1, 2, 3][Math.random() * 7 | 0];
  const p = () => spatialBeep(h2az(quiz.cur), 0, 900, 0.5); setTimeout(p, 700); setTimeout(p, 950); }
$('qz').onclick = async () => { if (!ac) initAudio(); await ac.resume(); qd.style.display = 'grid'; quiz.n = quiz.ok = 0; st.textContent = '어느 시 방향에서 들리나요? (이어폰 착용)'; next(); };
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
