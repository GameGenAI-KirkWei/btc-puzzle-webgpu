/**
 * Local WebGPU Bitcoin Puzzle Searcher — zero fee, keys never leave the browser.
 */
import * as secp from './lib/noble-secp256k1.js';
import {
  addressToHash160, hash160ToAddress, privateKeyToWIF, bytesToHex, hexToBytes,
  bigintTo32BE, bigintToLimbsLE, limbsLEToBigint, hash160ToWordsBE, bigintFrom32BE,
  sha256, hash160,
} from './lib/crypto-utils.js';

const LS_DISPATCH = 'btcPuzzle.webgpu.dispatchKeys';
const LS_FINGERPRINT = 'btcPuzzle.webgpu.selftest.fp';
const LS_SELFTEST_OK = 'btcPuzzle.webgpu.selftest.ok';

const PUZZLES = {
  demo: {
    id: 'demo',
    label: 'DEMO（快速自測：搵 key 5）',
    address: '17Vu7st1U1KwymUKU4jJheHHGRVNqrcfLD', // key 5; also resolved at runtime
    start: 1n,
    end: 32n,
    targetKey: 5n,
  },
  71: {
    id: '71',
    label: 'Puzzle #71（7.1 BTC）',
    address: '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU',
    start: 0x400000000000000000n,
    end: 0x7fffffffffffffffffn,
  },
  72: {
    id: '72',
    label: 'Puzzle #72（7.2 BTC）',
    address: '1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR',
    start: 0x800000000000000000n,
    end: 0xffffffffffffffffffn,
  },
  73: {
    id: '73',
    label: 'Puzzle #73（7.3 BTC）',
    address: '12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4',
    start: 0x1000000000000000000n,
    end: 0x1ffffffffffffffffffn,
  },
  custom: {
    id: 'custom',
    label: '自訂範圍',
    address: '',
    start: 1n,
    end: 100n,
  },
};

const VECTORS = {
  key1Addr: '1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH',
  key2Addr: '1cMh228HTCiwS8ZsaakH8A8wze1JR5ZsP',
  key3Addr: '1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb',
  puzzle71Addr: '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU',
  puzzle71Hash160: 'f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8',
};

/** @type {GPUDevice|null} */
let device = null;
/** @type {GPUAdapter|null} */
let adapter = null;
let shaderModule = null;
let pipeline = null;
let running = false;
let abortFlag = false;
let selftestPassed = false;
let totalChecked = 0n;
let keysPerSec = 0;
let dispatchKeys = 3072;
let currentChunk = '';
let findCallback = null;

const $ = (id) => document.getElementById(id);

function log(...a) { console.log('[puzzle]', ...a); }

async function resolveDemoAddress() {
  const pub = secp.getPublicKey(bigintTo32BE(5n), true);
  const h = await hash160(pub);
  const addr = await hash160ToAddress(h);
  PUZZLES.demo.address = addr;
  return addr;
}

function pointToLimbs(pubCompressed) {
  // noble getPublicKey compressed → need affine x,y as LE limbs
  const point = secp.ProjectivePoint.fromHex(pubCompressed);
  const { x, y } = { x: point.x, y: point.y };
  return { x: bigintToLimbsLE(x), y: bigintToLimbsLE(y) };
}

function affineFromKey(k) {
  const pub = secp.getPublicKey(bigintTo32BE(k), true);
  const point = secp.ProjectivePoint.fromHex(pub);
  return {
    x: bigintToLimbsLE(point.x),
    y: bigintToLimbsLE(point.y),
    compressed: pub,
  };
}

function packAff(xLimbs, yLimbs) {
  const u = new Uint32Array(16);
  u.set(xLimbs, 0);
  u.set(yLimbs, 8);
  return u;
}

async function initWebGPU() {
  if (!navigator.gpu) throw new Error('呢個瀏覽器冇 WebGPU（請用最新 Chrome / Edge）');
  adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('搵唔到 GPU adapter');
  device = await adapter.requestDevice();
  device.lost.then((info) => {
    log('device lost', info);
    pipeline = null;
    device = null;
  });
  const code = await (await fetch('./shaders/search.wgsl')).text();
  shaderModule = device.createShaderModule({ code });
  const info = await shaderModule.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) {
    console.error(errs);
    throw new Error('WGSL 編譯失敗: ' + errs.map(e => e.message).join('; '));
  }
  pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });
  const saved = localStorage.getItem(LS_DISPATCH);
  if (saved) dispatchKeys = Math.max(64, parseInt(saved, 10) || 3072);
  return {
    adapterName: adapter.info?.device || adapter.info?.description || 'WebGPU adapter',
    vendor: adapter.info?.vendor || '',
  };
}

function adapterFingerprint() {
  const info = adapter?.info || {};
  return JSON.stringify({
    v: info.vendor, d: info.device, a: info.architecture, s: 'search.wgsl.v1',
  });
}

function buildOffsetTable(keysPerThread) {
  // offset_table[bit] = (keysPerThread * 2^bit) * G as affine
  const table = new Uint32Array(256);
  for (let bit = 0; bit < 16; bit++) {
    const scalar = BigInt(keysPerThread) * (1n << BigInt(bit));
    const aff = affineFromKey(scalar % secp.CURVE.n);
    table.set(aff.x, bit * 16);
    table.set(aff.y, bit * 16 + 8);
  }
  return table;
}

/**
 * Run one GPU chunk: keys [start, start+count)
 * @returns {{found:boolean, key?:bigint, ms:number, count:number}}
 */
async function runChunk(startKey, count, targetWords) {
  if (!device || !pipeline) throw new Error('GPU 未就緒');

  // Choose thread_count and keys_per_thread
  let threadCount = 64;
  while (threadCount * 2 <= count && threadCount < 4096) threadCount *= 2;
  // fit count
  let keysPerThread = Math.ceil(count / threadCount);
  // align keysPerThread to BATCH=16
  keysPerThread = Math.max(48, Math.ceil(keysPerThread / 48) * 48);
  const actualCount = threadCount * keysPerThread;

  const base = affineFromKey(startKey);
  const G = affineFromKey(1n);
  const offsetTable = buildOffsetTable(keysPerThread);

  const uniformData = new ArrayBuffer(64);
  const u32 = new Uint32Array(uniformData);
  const sk = bigintToLimbsLE(startKey);
  u32[0] = targetWords[0]; u32[1] = targetWords[1]; u32[2] = targetWords[2];
  u32[3] = targetWords[3]; u32[4] = targetWords[4];
  u32[5] = keysPerThread;
  u32[6] = 0;
  u32[7] = threadCount;
  for (let i = 0; i < 8; i++) u32[8 + i] = sk[i];

  const uniformBuf = device.createBuffer({
    size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, uniformData);

  const baseBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(baseBuf, 0, packAff(base.x, base.y));

  const offBuf = device.createBuffer({ size: offsetTable.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(offBuf, 0, offsetTable);

  const gBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(gBuf, 0, packAff(G.x, G.y));

  const resultBuf = device.createBuffer({
    size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(resultBuf, 0, new Uint32Array(12));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: baseBuf } },
      { binding: 2, resource: { buffer: offBuf } },
      { binding: 3, resource: { buffer: gBuf } },
      { binding: 4, resource: { buffer: resultBuf } },
    ],
  });

  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(threadCount / 64));
  pass.end();

  const readBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyBufferToBuffer(resultBuf, 0, readBuf, 0, 48);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const ms = performance.now() - t0;
  const out = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  uniformBuf.destroy(); baseBuf.destroy(); offBuf.destroy(); gBuf.destroy();
  resultBuf.destroy(); readBuf.destroy();

  const found = out[0] === 1;
  let key;
  if (found) {
    key = limbsLEToBigint(out.subarray(1, 9));
  }
  return { found, key, ms, count: actualCount, threadCount, keysPerThread };
}

/** CPU path for DEMO / fallback verification */
async function cpuSearchRange(start, end, targetH160, onProgress) {
  const target = bytesToHex(targetH160);
  for (let k = start; k <= end; k++) {
    const pub = secp.getPublicKey(bigintTo32BE(k), true);
    const h = await hash160(pub);
    if (bytesToHex(h) === target) return k;
    if (onProgress && (k - start) % 8n === 0n) onProgress(k);
  }
  return null;
}

async function runSelfTests(ui) {
  const rows = [];
  const add = (name, pass, detail) => {
    rows.push({ name, pass, detail });
    if (ui) ui(rows);
  };

  // 1) Address decode puzzle 71
  try {
    const h = await addressToHash160(VECTORS.puzzle71Addr);
    const ok = bytesToHex(h) === VECTORS.puzzle71Hash160;
    add('Puzzle 71 HASH160 解碼', ok, bytesToHex(h));
  } catch (e) {
    add('Puzzle 71 HASH160 解碼', false, String(e));
  }

  // 2) CPU key1/2/3
  for (const [k, addr] of [[1n, VECTORS.key1Addr], [2n, VECTORS.key2Addr], [3n, VECTORS.key3Addr]]) {
    const pub = secp.getPublicKey(bigintTo32BE(k), true);
    const a = await hash160ToAddress(await hash160(pub));
    add(`CPU key ${k} → 地址`, a === addr, a);
  }

  // 3) +G from key1 → key2
  {
    const p1 = secp.ProjectivePoint.fromPrivateKey(bigintTo32BE(1n));
    const p2 = p1.add(secp.ProjectivePoint.BASE);
    const pub = p2.toRawBytes(true);
    const a = await hash160ToAddress(await hash160(pub));
    add('CPU key1 + G → key2 地址', a === VECTORS.key2Addr, a);
  }

  // 4) GPU tests if available
  if (device && pipeline) {
    const tests = [
      { name: 'GPU 搵 key 1', start: 1n, count: 64, addr: VECTORS.key1Addr, expect: 1n },
      { name: 'GPU 搵 key 3', start: 1n, count: 64, addr: VECTORS.key3Addr, expect: 3n },
      { name: 'GPU 由 1 行至 2', start: 1n, count: 64, addr: VECTORS.key2Addr, expect: 2n },
    ];
    for (const t of tests) {
      try {
        const h160 = await addressToHash160(t.addr);
        const words = hash160ToWordsBE(h160);
        const r = await runChunk(t.start, t.count, words);
        const ok = r.found && r.key === t.expect;
        add(t.name, ok, r.found ? `key=${r.key} (${r.ms.toFixed(1)}ms)` : `未命中 (${r.ms.toFixed(1)}ms)`);
      } catch (e) {
        add(t.name, false, String(e.message || e));
      }
    }
  } else {
    add('GPU 自測', false, 'WebGPU 未可用 — 只能用 CPU DEMO');
  }

  selftestPassed = rows.every(r => r.pass || (r.name === 'GPU 自測' && !device));
  // Require all that ran: if GPU present, GPU tests must pass
  selftestPassed = rows.every(r => r.pass);
  if (selftestPassed) {
    localStorage.setItem(LS_FINGERPRINT, adapterFingerprint());
    localStorage.setItem(LS_SELFTEST_OK, '1');
  }
  return rows;
}

function randomStartInRange(start, end, span) {
  const width = end - start + 1n;
  if (span >= width) return start;
  const maxStart = width - span;
  // random bigint
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  let r = 0n;
  for (const b of bytes) r = (r << 8n) | BigInt(b);
  return start + (r % (maxStart + 1n));
}

function medianEta(keysLeft, rate) {
  if (rate <= 0) return '—';
  // geometric: expected half the space for random with replacement approx keysLeft/2 / rate
  const sec = Number(keysLeft / 2n) / rate;
  if (!isFinite(sec) || sec > 1e15) return '遠超宇宙壽命';
  const years = sec / (365.25 * 24 * 3600);
  if (years > 1e6) return years.toExponential(2) + ' 年（中位數估計）';
  if (years > 1) return years.toFixed(1) + ' 年（中位數估計）';
  if (sec > 3600) return (sec / 3600).toFixed(1) + ' 小時';
  if (sec > 60) return (sec / 60).toFixed(1) + ' 分鐘';
  return sec.toFixed(1) + ' 秒';
}

async function onFound(key, address) {
  running = false;
  const priv = bigintTo32BE(key);
  const wif = await privateKeyToWIF(priv);
  const hex = bytesToHex(priv);
  $('hit-panel').hidden = false;
  $('hit-hex').textContent = hex;
  $('hit-wif').textContent = wif;
  $('hit-addr').textContent = address;
  $('hit-panel').scrollIntoView({ behavior: 'smooth' });
}

async function searchLoop(config) {
  const h160 = await addressToHash160(config.address);
  const words = hash160ToWordsBE(h160);
  const rangeSize = config.end - config.start + 1n;
  totalChecked = 0n;
  abortFlag = false;
  running = true;
  let windowMs = [];
  let rate = 0;

  while (running && !abortFlag) {
    if (!device || !pipeline) {
      // try reacquire
      try {
        await initWebGPU();
        dispatchKeys = Math.max(64, Math.floor(dispatchKeys * 0.5));
      } catch (e) {
        $('status').textContent = 'GPU 丢失，重連失敗: ' + e.message;
        break;
      }
    }

    const span = BigInt(dispatchKeys);
    const start = randomStartInRange(config.start, config.end, span);
    const count = Number(span > rangeSize ? rangeSize : span);
    currentChunk = `0x${start.toString(16)} … +${count}`;
    $('chunk').textContent = currentChunk;

    let result;
    try {
      result = await runChunk(start, count, words);
    } catch (e) {
      log('chunk error', e);
      dispatchKeys = Math.max(64, Math.floor(dispatchKeys / 2));
      localStorage.setItem(LS_DISPATCH, String(dispatchKeys));
      pipeline = null;
      continue;
    }

    totalChecked += BigInt(result.count);
    windowMs.push({ ms: result.ms, n: result.count });
    if (windowMs.length > 8) windowMs.shift();
    const sumN = windowMs.reduce((a, b) => a + b.n, 0);
    const sumT = windowMs.reduce((a, b) => a + b.ms, 0) / 1000;
    rate = sumT > 0 ? sumN / sumT : 0;
    keysPerSec = rate;

    // calibrate toward ~250ms
    if (result.ms > 500) {
      dispatchKeys = Math.max(64, Math.floor(dispatchKeys * 0.6));
      localStorage.setItem(LS_DISPATCH, String(dispatchKeys));
    } else if (result.ms < 120 && dispatchKeys < 1 << 22) {
      dispatchKeys = Math.floor(dispatchKeys * 1.4);
      localStorage.setItem(LS_DISPATCH, String(dispatchKeys));
    }

    $('kps').textContent = formatRate(rate);
    $('checked').textContent = totalChecked.toLocaleString();
    $('dispatch').textContent = String(dispatchKeys);
    $('eta').textContent = medianEta(rangeSize, rate);

    if (result.found) {
      // verify on CPU
      const pub = secp.getPublicKey(bigintTo32BE(result.key), true);
      const addr = await hash160ToAddress(await hash160(pub));
      if (addr === config.address) {
        await onFound(result.key, addr);
        return;
      } else {
        log('false positive?', result.key, addr);
      }
    }
    await new Promise(r => setTimeout(r, 0));
  }
  running = false;
  $('btn-start').disabled = !selftestPassed;
  $('btn-stop').disabled = true;
  $('status').textContent = abortFlag ? '已停止' : '已結束';
}

function formatRate(r) {
  if (r > 1e9) return (r / 1e9).toFixed(2) + ' Gkeys/s';
  if (r > 1e6) return (r / 1e6).toFixed(2) + ' Mkeys/s';
  if (r > 1e3) return (r / 1e3).toFixed(2) + ' kkeys/s';
  return r.toFixed(0) + ' keys/s';
}

async function cpuDemoSearch(uiStatus) {
  const addr = PUZZLES.demo.address || await resolveDemoAddress();
  const h160 = await addressToHash160(addr);
  uiStatus('CPU DEMO 搜緊 key 5…');
  const t0 = performance.now();
  const key = await cpuSearchRange(1n, 32n, h160, () => {});
  const ms = performance.now() - t0;
  if (key != null) {
    await onFound(key, addr);
    uiStatus(`CPU DEMO 命中 key ${key}（${ms.toFixed(1)} ms）`);
  } else {
    uiStatus('CPU DEMO 未命中（不應發生）');
  }
}

function getConfigFromUI() {
  const id = $('puzzle').value;
  if (id === 'custom') {
    const address = $('custom-addr').value.trim();
    const start = BigInt('0x' + $('custom-start').value.trim().replace(/^0x/i, ''));
    const end = BigInt('0x' + $('custom-end').value.trim().replace(/^0x/i, ''));
    if (!address) throw new Error('請填地址');
    if (end < start) throw new Error('範圍結束要 ≥ 開始');
    return { address, start, end, id };
  }
  const p = PUZZLES[id];
  return { address: p.address, start: p.start, end: p.end, id };
}

function oddsNote(id) {
  if (id === '71') {
    return 'Puzzle 71 有 2^70 ≈ 1.18×10²¹ 個候選私鑰。以瀏覽器 GPU（例如 5×10⁷ keys/s）亂撞，中位數時間以「億年」計——本質上係公開彩票，唔係穩賺工具。';
  }
  if (id === 'demo') return 'DEMO 範圍極細，只用嚟驗證流程同命中 UI。';
  return '未解謎題範圍仍然極大；請以娛樂／研究心態對待。';
}

async function main() {
  await resolveDemoAddress();
  // fill known demo address into select label already

  $('puzzle').innerHTML = Object.values(PUZZLES).map(p =>
    `<option value="${p.id}">${p.label}</option>`
  ).join('');
  $('puzzle').value = '71';

  const syncCustom = () => {
    const custom = $('puzzle').value === 'custom';
    $('custom-fields').hidden = !custom;
    $('odds').textContent = oddsNote($('puzzle').value);
  };
  $('puzzle').addEventListener('change', syncCustom);
  syncCustom();

  $('btn-copy-hex').onclick = () => navigator.clipboard.writeText($('hit-hex').textContent);
  $('btn-copy-wif').onclick = () => navigator.clipboard.writeText($('hit-wif').textContent);

  $('btn-stop').onclick = () => { abortFlag = true; running = false; };
  $('btn-start').disabled = true;
  $('btn-stop').disabled = true;

  $('btn-cpu-demo').onclick = async () => {
    try {
      await cpuDemoSearch((t) => { $('status').textContent = t; });
    } catch (e) {
      $('status').textContent = String(e.message || e);
    }
  };

  $('status').textContent = '初始化 WebGPU…';
  let gpuInfo = { adapterName: 'N/A' };
  try {
    gpuInfo = await initWebGPU();
    $('gpu').textContent = `${gpuInfo.adapterName} / ${navigator.userAgent.match(/Chrome\/[\d.]+|Edg\/[\d.]+/)?.[0] || 'browser'}`;
    $('status').textContent = 'WebGPU 就緒，跑緊自測…';
  } catch (e) {
    $('gpu').textContent = '不可用: ' + e.message;
    $('status').textContent = 'WebGPU 失敗——你仍然可以用 CPU DEMO 自測命中 UI。';
  }

  const fp = adapterFingerprint();
  const cached = localStorage.getItem(LS_SELFTEST_OK) === '1' && localStorage.getItem(LS_FINGERPRINT) === fp;

  const renderTests = (rows) => {
    const tb = $('selftest-body');
    tb.innerHTML = rows.map(r =>
      `<tr class="${r.pass ? 'pass' : 'fail'}"><td>${r.name}</td><td>${r.pass ? '✅ 通過' : '❌ 失敗'}</td><td><code>${r.detail}</code></td></tr>`
    ).join('');
  };

  if (cached && device) {
    $('status').textContent = '自測指紋未變，略過重測（可撳「強制重測」）';
    selftestPassed = true;
    renderTests([{ name: '快取自測', pass: true, detail: 'fingerprint match' }]);
  } else {
    const rows = await runSelfTests(renderTests);
    renderTests(rows);
    $('status').textContent = selftestPassed ? '自測全部通過，可以開始' : '自測未全部通過 — Start 已鎖';
  }

  $('btn-retest').onclick = async () => {
    localStorage.removeItem(LS_SELFTEST_OK);
    $('status').textContent = '強制重測…';
    const rows = await runSelfTests(renderTests);
    renderTests(rows);
    $('btn-start').disabled = !selftestPassed || running;
    $('status').textContent = selftestPassed ? '自測全部通過' : '自測失敗';
  };

  $('btn-start').disabled = !selftestPassed;
  $('dispatch').textContent = String(dispatchKeys);

  $('btn-start').onclick = async () => {
    if (!selftestPassed) return;
    let config;
    try { config = getConfigFromUI(); }
    catch (e) { alert(e.message); return; }

    // For demo without GPU, use CPU
    if (config.id === 'demo' && !device) {
      await cpuDemoSearch((t) => { $('status').textContent = t; });
      return;
    }
    if (!device) {
      alert('需要 WebGPU 先可以搜正式謎題');
      return;
    }

    $('hit-panel').hidden = true;
    $('btn-start').disabled = true;
    $('btn-stop').disabled = false;
    $('status').textContent = '搜尋中（純本地、零抽成）…';
    try {
      await searchLoop(config);
    } catch (e) {
      console.error(e);
      $('status').textContent = '錯誤: ' + e.message;
    }
    $('btn-start').disabled = !selftestPassed;
    $('btn-stop').disabled = true;
  };
}

main().catch((e) => {
  console.error(e);
  const s = document.getElementById('status');
  if (s) s.textContent = '啟動失敗: ' + e.message;
});
