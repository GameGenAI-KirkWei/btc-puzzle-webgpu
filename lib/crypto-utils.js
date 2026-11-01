/** Base58Check, WIF, address helpers. ESM. */
import { ripemd160 } from './ripemd160.js';

const ALPH = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';

export function bytesToHex(bytes) {
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}
export function hexToBytes(hex) {
  const h = hex.replace(/^0x/i, '').replace(/\s+/g, '');
  if (h.length % 2) throw new Error('odd hex length');
  const out = new Uint8Array(h.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(h.slice(i * 2, i * 2 + 2), 16);
  return out;
}

export async function sha256(data) {
  const buf = await crypto.subtle.digest('SHA-256', data);
  return new Uint8Array(buf);
}

export async function hash160(data) {
  return ripemd160(await sha256(data));
}

export function base58Encode(bytes) {
  let zeros = 0;
  while (zeros < bytes.length && bytes[zeros] === 0) zeros++;
  // count digits needed
  const size = Math.ceil(bytes.length * 138 / 100) + 1;
  const b = new Uint8Array(size);
  for (const byte of bytes) {
    let carry = byte;
    for (let j = size - 1; j >= 0; j--) {
      carry += b[j] << 8;
      b[j] = carry % 58;
      carry = (carry / 58) | 0;
    }
  }
  let i = 0;
  while (i < size && b[i] === 0) i++;
  let out = '1'.repeat(zeros);
  for (; i < size; i++) out += ALPH[b[i]];
  return out;
}

export function base58Decode(str) {
  let zeros = 0;
  while (zeros < str.length && str[zeros] === '1') zeros++;
  const size = Math.ceil(str.length * 733 / 1000) + 1;
  const b = new Uint8Array(size);
  for (const ch of str) {
    const val = ALPH.indexOf(ch);
    if (val < 0) throw new Error('invalid base58');
    let carry = val;
    for (let j = size - 1; j >= 0; j--) {
      carry += b[j] * 58;
      b[j] = carry & 0xff;
      carry >>= 8;
    }
  }
  let i = 0;
  while (i < size && b[i] === 0) i++;
  const out = new Uint8Array(zeros + (size - i));
  out.fill(0, 0, zeros);
  out.set(b.subarray(i), zeros);
  return out;
}

export async function base58CheckEncode(payload) {
  const cs = (await sha256(await sha256(payload))).subarray(0, 4);
  const full = new Uint8Array(payload.length + 4);
  full.set(payload); full.set(cs, payload.length);
  return base58Encode(full);
}

export async function base58CheckDecode(addr) {
  const raw = base58Decode(addr);
  if (raw.length < 5) throw new Error('too short');
  const payload = raw.subarray(0, raw.length - 4);
  const cs = raw.subarray(raw.length - 4);
  const expect = (await sha256(await sha256(payload))).subarray(0, 4);
  for (let i = 0; i < 4; i++) if (cs[i] !== expect[i]) throw new Error('bad checksum');
  return payload;
}

/** Decode P2PKH address → 20-byte HASH160 */
export async function addressToHash160(addr) {
  const payload = await base58CheckDecode(addr);
  if (payload.length !== 21 || payload[0] !== 0x00) throw new Error('只支援 P2PKH (1...) 地址');
  return payload.subarray(1);
}

export async function hash160ToAddress(h160) {
  const payload = new Uint8Array(21);
  payload[0] = 0x00;
  payload.set(h160, 1);
  return base58CheckEncode(payload);
}

/** Compressed WIF for mainnet */
export async function privateKeyToWIF(priv32) {
  if (priv32.length !== 32) throw new Error('priv must be 32 bytes');
  const payload = new Uint8Array(34);
  payload[0] = 0x80;
  payload.set(priv32, 1);
  payload[33] = 0x01;
  return base58CheckEncode(payload);
}

export function u8ToBeHex(bytes) {
  return bytesToHex(bytes);
}

/** Big-endian 32-byte from BigInt */
export function bigintTo32BE(n) {
  const out = new Uint8Array(32);
  let x = BigInt(n);
  for (let i = 31; i >= 0; i--) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return out;
}

export function bigintFrom32BE(bytes) {
  let x = 0n;
  for (const b of bytes) x = (x << 8n) | BigInt(b);
  return x;
}

/** LE u32 limbs (8) from BigInt */
export function bigintToLimbsLE(n, len = 8) {
  const out = new Uint32Array(len);
  let x = BigInt(n);
  for (let i = 0; i < len; i++) {
    out[i] = Number(x & 0xffffffffn);
    x >>= 32n;
  }
  return out;
}

export function limbsLEToBigint(limbs) {
  let x = 0n;
  for (let i = limbs.length - 1; i >= 0; i--) x = (x << 32n) | BigInt(limbs[i] >>> 0);
  return x;
}

/** Pack 20-byte hash160 into 5 u32 big-endian words (for easy WGSL byte compare) */
export function hash160ToWordsBE(h160) {
  const w = new Uint32Array(5);
  const dv = new DataView(h160.buffer, h160.byteOffset, 20);
  for (let i = 0; i < 5; i++) w[i] = dv.getUint32(i * 4, false);
  return w;
}
