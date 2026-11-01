// Local WebGPU Bitcoin Puzzle searcher — WGSL
// secp256k1 Jacobian walk + SHA-256 + RIPEMD-160

alias Fe = array<u32, 8>;

struct Uniforms {
  target0: u32, target1: u32, target2: u32, target3: u32, target4: u32,
  keys_per_thread: u32,
  _pad0: u32,
  thread_count: u32,
  sk0: u32, sk1: u32, sk2: u32, sk3: u32,
  sk4: u32, sk5: u32, sk6: u32, sk7: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> base_xy: array<u32, 16>;
@group(0) @binding(2) var<storage, read> offset_table: array<u32, 256>;
@group(0) @binding(3) var<storage, read> g_xy: array<u32, 16>;
@group(0) @binding(4) var<storage, read_write> result: array<atomic<u32>, 12>;

const BATCH: u32 = 48u;
const FE_P: Fe = Fe(4294966319u, 4294967294u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u);
const FE_ONE: Fe = Fe(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
const FE_ZERO: Fe = Fe(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
const FE_R: u32 = 0x000003d1u; // low part of 2^32+977 for reduction helpers

fn mul32(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu; let a1 = a >> 16u;
  let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  let lo = (p00 & 0xffffu) | (mid << 16u);
  let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
  return vec2<u32>(lo, hi);
}

fn fe_is_zero(a: Fe) -> bool {
  return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0u;
}

fn fe_add(a: Fe, b: Fe) -> Fe {
  var r: Fe;
  var carry: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i++) {
    let s = a[i] + carry;
    let c1 = select(0u, 1u, s < a[i]);
    let t = s + b[i];
    let c2 = select(0u, 1u, t < s);
    r[i] = t;
    carry = c1 + c2;
  }
  var ge = (carry != 0u);
  if (!ge) {
    var decided = false;
    for (var i: i32 = 7; i >= 0; i--) {
      if (!decided) {
        let ai = r[u32(i)]; let pi = FE_P[u32(i)];
        if (ai != pi) { ge = ai > pi; decided = true; }
      }
    }
    if (!decided) { ge = true; } // equal
  }
  if (ge) {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i++) {
      let tmp = r[i] - borrow;
      let b1 = select(0u, 1u, r[i] < borrow);
      let res = tmp - FE_P[i];
      let b2 = select(0u, 1u, tmp < FE_P[i]);
      r[i] = res;
      borrow = b1 | b2;
    }
  }
  return r;
}

fn fe_sub(a: Fe, b: Fe) -> Fe {
  var r: Fe;
  var borrow: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i++) {
    let tmp = a[i] - borrow;
    let b1 = select(0u, 1u, a[i] < borrow);
    let res = tmp - b[i];
    let b2 = select(0u, 1u, tmp < b[i]);
    r[i] = res;
    borrow = b1 | b2;
  }
  if (borrow != 0u) {
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i++) {
      let s = r[i] + carry;
      let c1 = select(0u, 1u, s < r[i]);
      let t = s + FE_P[i];
      let c2 = select(0u, 1u, t < s);
      r[i] = t;
      carry = c1 + c2;
    }
  }
  return r;
}

fn fe_neg(a: Fe) -> Fe {
  if (fe_is_zero(a)) { return FE_ZERO; }
  return fe_sub(FE_P, a);
}

fn add_u32(a: u32, b: u32, cin: u32) -> vec2<u32> {
  let s0 = a + b;
  let c0 = select(0u, 1u, s0 < a);
  let s1 = s0 + cin;
  let c1 = select(0u, 1u, s1 < s0);
  return vec2<u32>(s1, c0 + c1);
}

fn fe_mul(a: Fe, b: Fe) -> Fe {
  var p: array<u32, 16>;
  for (var i: u32 = 0u; i < 16u; i++) { p[i] = 0u; }

  for (var i: u32 = 0u; i < 8u; i++) {
    var carry: u32 = 0u;
    for (var j: u32 = 0u; j < 8u; j++) {
      let m = mul32(a[i], b[j]);
      let r0 = add_u32(p[i + j], m.x, carry);
      p[i + j] = r0.x;
      let r1 = add_u32(m.y, r0.y, 0u);
      carry = r1.x;
      if (r1.y != 0u) {
        let idx = i + j + 1u;
        let r2 = add_u32(p[idx], r1.y, 0u);
        p[idx] = r2.x;
        carry = carry + r2.y;
      }
    }
    var k = i + 8u;
    while (carry != 0u && k < 16u) {
      let r = add_u32(p[k], carry, 0u);
      p[k] = r.x;
      carry = r.y;
      k += 1u;
    }
  }
  return fe_reduce512(p);
}

fn fe_reduce512(p_in: array<u32, 16>) -> Fe {
  var p = p_in;
  for (var round: u32 = 0u; round < 4u; round++) {
    var hi_nonzero = false;
    for (var i: u32 = 8u; i < 16u; i++) {
      if (p[i] != 0u) { hi_nonzero = true; }
    }
    if (!hi_nonzero) { break; }

    var mid: array<u32, 16>;
    for (var i: u32 = 0u; i < 16u; i++) { mid[i] = 0u; }

    for (var i: u32 = 0u; i < 8u; i++) {
      let m = mul32(p[8u + i], 977u);
      let r0 = add_u32(mid[i], m.x, 0u);
      mid[i] = r0.x;
      let r1 = add_u32(mid[i+1u], m.y, r0.y);
      mid[i+1u] = r1.x;
      var k = i + 2u;
      var c = r1.y;
      while (c != 0u && k < 16u) {
        let r = add_u32(mid[k], c, 0u); mid[k] = r.x; c = r.y; k += 1u;
      }
    }
    for (var i: u32 = 0u; i < 8u; i++) {
      let r0 = add_u32(mid[i+1u], p[8u + i], 0u);
      mid[i+1u] = r0.x;
      var k = i + 2u;
      var c = r0.y;
      while (c != 0u && k < 16u) {
        let r = add_u32(mid[k], c, 0u); mid[k] = r.x; c = r.y; k += 1u;
      }
    }
    var lo: array<u32, 16>;
    for (var i: u32 = 0u; i < 8u; i++) { lo[i] = p[i]; }
    for (var i: u32 = 8u; i < 16u; i++) { lo[i] = 0u; }
    var c: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i++) {
      let r = add_u32(lo[i], mid[i], c);
      p[i] = r.x; c = r.y;
    }
  }

  var r: Fe;
  for (var i: u32 = 0u; i < 8u; i++) { r[i] = p[i]; }

  for (var n: u32 = 0u; n < 4u; n++) {
    var ge = false;
    var decided = false;
    for (var i: i32 = 7; i >= 0; i--) {
      if (!decided) {
        let ai = r[u32(i)]; let pi = FE_P[u32(i)];
        if (ai != pi) { ge = ai > pi; decided = true; }
      }
    }
    if (!decided) { ge = true; }
    if (ge) {
      var borrow: u32 = 0u;
      for (var i: u32 = 0u; i < 8u; i++) {
        let tmp = r[i] - borrow;
        let b1 = select(0u, 1u, r[i] < borrow);
        let res = tmp - FE_P[i];
        let b2 = select(0u, 1u, tmp < FE_P[i]);
        r[i] = res;
        borrow = b1 | b2;
      }
    } else { break; }
  }
  return r;
}

fn fe_sqr(a: Fe) -> Fe { return fe_mul(a, a); }

fn fe_inv(a: Fe) -> Fe {
  // Fermat: a^(p-2) mod p. Binary exponentiation MSB-first on (p-2).
  // p-2 limbs LE: FC2D, FFFE, FFFF, FFFF, FFFF, FFFF, FFFF, FFFF
  let e = Fe(0xfffffc2du, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
             0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  var base = a;
  var res = FE_ONE;
  for (var i: u32 = 0u; i < 8u; i++) {
    var limb = e[i];
    for (var b: u32 = 0u; b < 32u; b++) {
      if ((limb & 1u) != 0u) { res = fe_mul(res, base); }
      base = fe_sqr(base);
      limb = limb >> 1u;
    }
  }
  return res;
}

// Batch Montgomery inversion: inv[i] = 1/z[i]
fn batch_invert(z: ptr<function, array<Fe, 48>>, inv: ptr<function, array<Fe, 48>>, n: u32) {
  var acc: array<Fe, 48>;
  acc[0] = (*z)[0];
  for (var i: u32 = 1u; i < n; i++) {
    acc[i] = fe_mul(acc[i - 1u], (*z)[i]);
  }
  var inv_all = fe_inv(acc[n - 1u]);
  for (var i: i32 = i32(n) - 1; i > 0; i--) {
    (*inv)[u32(i)] = fe_mul(acc[u32(i) - 1u], inv_all);
    inv_all = fe_mul(inv_all, (*z)[u32(i)]);
  }
  (*inv)[0] = inv_all;
}


struct Jac { x: Fe, y: Fe, z: Fe, };
struct Aff { x: Fe, y: Fe, };

fn load_offset_aff(bit: u32) -> Aff {
  var a: Aff;
  let buf_offset_u32 = bit * 16u;
  for (var i: u32 = 0u; i < 8u; i++) { a.x[i] = offset_table[buf_offset_u32 + i]; }
  for (var i: u32 = 0u; i < 8u; i++) { a.y[i] = offset_table[buf_offset_u32 + 8u + i]; }
  return a;
}

fn load_base() -> Aff {
  var a: Aff;
  for (var i: u32 = 0u; i < 8u; i++) { a.x[i] = base_xy[i]; }
  for (var i: u32 = 0u; i < 8u; i++) { a.y[i] = base_xy[8u + i]; }
  return a;
}

fn load_G() -> Aff {
  var a: Aff;
  for (var i: u32 = 0u; i < 8u; i++) { a.x[i] = g_xy[i]; }
  for (var i: u32 = 0u; i < 8u; i++) { a.y[i] = g_xy[8u + i]; }
  return a;
}

fn jac_from_aff(a: Aff) -> Jac {
  return Jac(a.x, a.y, FE_ONE);
}

// Mixed add: Jacobian + Affine (assume a is not infinity, P not infinity, P != ±Q)
fn jac_is_inf(P: Jac) -> bool { return fe_is_zero(P.z); }

fn jac_double(P: Jac) -> Jac {
  // secp256k1 a=0 doubling
  if (fe_is_zero(P.z)) {
    return Jac(FE_ZERO, FE_ONE, FE_ZERO);
  }
  let A = fe_sqr(P.x);
  let B = fe_sqr(P.y);
  let C = fe_sqr(B);
  // D = 2*((X+B)^2 - A - C) = 4*X*B
  let D = fe_sub(fe_sqr(fe_add(P.x, B)), fe_add(A, C));
  let D2 = fe_add(D, D);
  let E = fe_add(fe_add(A, A), A); // 3*X^2
  let F = fe_sqr(E);
  let X3 = fe_sub(F, fe_add(D2, D2));
  let C2 = fe_add(C, C);
  let C4 = fe_add(C2, C2);
  let C8 = fe_add(C4, C4);
  let Y3 = fe_sub(fe_mul(E, fe_sub(D2, X3)), C8);
  let Z3 = fe_mul(fe_add(P.y, P.y), P.z);
  return Jac(X3, Y3, Z3);
}

fn jac_add_aff(P: Jac, Q: Aff) -> Jac {
  if (fe_is_zero(P.z)) {
    return Jac(Q.x, Q.y, FE_ONE);
  }
  // H = X2*Z1^2 - X1; R = Y2*Z1^3 - Y1
  let z1z1 = fe_sqr(P.z);
  let u2 = fe_mul(Q.x, z1z1);
  let s2 = fe_mul(Q.y, fe_mul(z1z1, P.z));
  let h = fe_sub(u2, P.x);
  let r = fe_sub(s2, P.y);
  if (fe_is_zero(h)) {
    if (fe_is_zero(r)) {
      return jac_double(P); // P == Q
    }
    return Jac(FE_ZERO, FE_ONE, FE_ZERO); // P == -Q → infinity
  }
  let hh = fe_sqr(h);
  let hhh = fe_mul(hh, h);
  let v = fe_mul(P.x, hh);
  let x3 = fe_sub(fe_sub(fe_sqr(r), hhh), fe_add(v, v));
  let y3 = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_mul(P.y, hhh));
  let z3 = fe_mul(P.z, h);
  return Jac(x3, y3, z3);
}

fn to_affine(P: Jac, zinv: Fe) -> Aff {
  let zinv2 = fe_sqr(zinv);
  let zinv3 = fe_mul(zinv2, zinv);
  return Aff(fe_mul(P.x, zinv2), fe_mul(P.y, zinv3));
}

const SHA_K: array<u32, 64> = array<u32, 64>(1116352408u, 1899447441u, 3049323471u, 3921009573u, 961987163u, 1508970993u, 2453635748u, 2870763221u, 3624381080u, 310598401u, 607225278u, 1426881987u, 1925078388u, 2162078206u, 2614888103u, 3248222580u, 3835390401u, 4022224774u, 264347078u, 604807628u, 770255983u, 1249150122u, 1555081692u, 1996064986u, 2554220882u, 2821834349u, 2952996808u, 3210313671u, 3336571891u, 3584528711u, 113926993u, 338241895u, 666307205u, 773529912u, 1294757372u, 1396182291u, 1695183700u, 1986661051u, 2177026350u, 2456956037u, 2730485921u, 2820302411u, 3259730800u, 3345764771u, 3516065817u, 3600352804u, 4094571909u, 275423344u, 430227734u, 506948616u, 659060556u, 883997877u, 958139571u, 1322822218u, 1537002063u, 1747873779u, 1955562222u, 2024104815u, 2227730452u, 2361852424u, 2428436474u, 2756734187u, 3204031479u, 3329325298u);

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn rotl(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }

fn sha256_compress(state: ptr<function, array<u32, 8>>, block: ptr<function, array<u32, 16>>) {
  var w: array<u32, 64>;
  for (var i: u32 = 0u; i < 16u; i++) { w[i] = (*block)[i]; }
  for (var i: u32 = 16u; i < 64u; i++) {
    let s0 = rotr(w[i-15u],7u) ^ rotr(w[i-15u],18u) ^ (w[i-15u] >> 3u);
    let s1 = rotr(w[i-2u],17u) ^ rotr(w[i-2u],19u) ^ (w[i-2u] >> 10u);
    w[i] = w[i-16u] + s0 + w[i-7u] + s1;
  }
  var a=(*state)[0]; var b=(*state)[1]; var c=(*state)[2]; var d=(*state)[3];
  var e=(*state)[4]; var f=(*state)[5]; var g=(*state)[6]; var h=(*state)[7];
  for (var i: u32 = 0u; i < 64u; i++) {
    let S1 = rotr(e,6u)^rotr(e,11u)^rotr(e,25u);
    let ch = (e & f) ^ ((~e) & g);
    let t1 = h + S1 + ch + SHA_K[i] + w[i];
    let S0 = rotr(a,2u)^rotr(a,13u)^rotr(a,22u);
    let maj = (a & b) ^ (a & c) ^ (b & c);
    let t2 = S0 + maj;
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  (*state)[0]+=a; (*state)[1]+=b; (*state)[2]+=c; (*state)[3]+=d;
  (*state)[4]+=e; (*state)[5]+=f; (*state)[6]+=g; (*state)[7]+=h;
}

// Hash 33-byte compressed pubkey → 32-byte SHA256 digest as 8 BE words
fn sha256_pubkey33(prefix: u32, x: Fe) -> array<u32, 8> {
  // Build 64-byte padded block as 16 BE words
  var block: array<u32, 16>;
  for (var i: u32 = 0u; i < 16u; i++) { block[i] = 0u; }
  // bytes: prefix || x_be[32] || 0x80 || zeros || len=264 bits
  // x limbs are LE; serialize X as big-endian bytes
  // word0: prefix<<24 | x_byte0<<16 | x_byte1<<8 | x_byte2
  // X bytes BE from limbs: limb7 has MSB
  var xb: array<u32, 32>;
  for (var i: u32 = 0u; i < 8u; i++) {
    let limb = x[7u - i];
    xb[i*4u+0u] = (limb >> 24u) & 0xffu;
    xb[i*4u+1u] = (limb >> 16u) & 0xffu;
    xb[i*4u+2u] = (limb >> 8u) & 0xffu;
    xb[i*4u+3u] = limb & 0xffu;
  }
  var msg: array<u32, 64>;
  msg[0] = prefix & 0xffu;
  for (var i: u32 = 0u; i < 32u; i++) { msg[1u+i] = xb[i]; }
  msg[33] = 0x80u;
  for (var i: u32 = 34u; i < 64u; i++) { msg[i] = 0u; }
  // length 33*8=264 = 0x108
  msg[62] = 0x01u; msg[63] = 0x08u;

  for (var i: u32 = 0u; i < 16u; i++) {
    block[i] = (msg[i*4u] << 24u) | (msg[i*4u+1u] << 16u) | (msg[i*4u+2u] << 8u) | msg[i*4u+3u];
  }
  var state: array<u32, 8> = array<u32, 8>(
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
  sha256_compress(&state, &block);
  return state;
}


const KL: array<u32, 5> = array<u32, 5>(0x00000000u,0x5A827999u,0x6ED9EBA1u,0x8F1BBCDCu,0xA953FD4Eu);
const KR: array<u32, 5> = array<u32, 5>(0x50A28BE6u,0x5C4DD124u,0x6D703EF3u,0x7A6D76E9u,0x00000000u);
const ZL: array<u32, 80> = array<u32, 80>(
 0u,1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,
 7u,4u,13u,1u,10u,6u,15u,3u,12u,0u,9u,5u,2u,14u,11u,8u,
 3u,10u,14u,4u,9u,15u,8u,1u,2u,7u,0u,6u,13u,11u,5u,12u,
 1u,9u,11u,10u,0u,8u,12u,4u,13u,3u,7u,15u,14u,5u,6u,2u,
 4u,0u,5u,9u,7u,12u,2u,10u,14u,1u,3u,8u,11u,6u,15u,13u);
const ZR: array<u32, 80> = array<u32, 80>(
 5u,14u,7u,0u,9u,2u,11u,4u,13u,6u,15u,8u,1u,10u,3u,12u,
 6u,11u,3u,7u,0u,13u,5u,10u,14u,15u,8u,12u,4u,9u,1u,2u,
 15u,5u,1u,3u,7u,14u,6u,9u,11u,8u,12u,2u,10u,0u,4u,13u,
 8u,6u,4u,1u,3u,11u,15u,0u,5u,12u,2u,13u,9u,7u,10u,14u,
 12u,15u,10u,4u,1u,5u,8u,7u,6u,2u,13u,14u,0u,3u,9u,11u);
const SL: array<u32, 80> = array<u32, 80>(
 11u,14u,15u,12u,5u,8u,7u,9u,11u,13u,14u,15u,6u,7u,9u,8u,
 7u,6u,8u,13u,11u,9u,7u,15u,7u,12u,15u,9u,11u,7u,13u,12u,
 11u,13u,6u,7u,14u,9u,13u,15u,14u,8u,13u,6u,5u,12u,7u,5u,
 11u,12u,14u,15u,14u,15u,9u,8u,9u,14u,5u,6u,8u,6u,5u,12u,
 9u,15u,5u,11u,6u,8u,13u,12u,5u,12u,13u,14u,11u,8u,5u,6u);
const SR: array<u32, 80> = array<u32, 80>(
 8u,9u,9u,11u,13u,15u,15u,5u,7u,7u,8u,11u,14u,14u,12u,6u,
 9u,13u,15u,7u,12u,8u,9u,11u,7u,7u,12u,7u,6u,15u,13u,11u,
 9u,7u,15u,11u,8u,6u,6u,14u,12u,13u,5u,14u,13u,13u,7u,5u,
 15u,5u,8u,11u,14u,14u,6u,14u,6u,9u,12u,9u,12u,5u,15u,8u,
 8u,5u,12u,9u,12u,5u,14u,6u,8u,13u,6u,5u,15u,13u,11u,11u);

fn ripemf(j: u32, x: u32, y: u32, z: u32) -> u32 {
  if (j < 16u) { return x ^ y ^ z; }
  if (j < 32u) { return (x & y) | ((~x) & z); }
  if (j < 48u) { return (x | (~y)) ^ z; }
  if (j < 64u) { return (x & z) | (y & (~z)); }
  return x ^ (y | (~z));
}

// SHA256 digest as 8 BE words → RIPEMD160 → 5 LE words (standard RIPEMD output endian)
fn ripemd160_sha_words(sw: array<u32, 8>) -> array<u32, 5> {
  // Message is 32 bytes of SHA digest (BE words), pad to 64
  var w: array<u32, 16>;
  // RIPEMD uses LE words: convert each BE sha word to bytes then LE word
  for (var i: u32 = 0u; i < 8u; i++) {
    let be = sw[i];
    // bytes b0..b3 as BE from be; store as LE word
    w[i] = ((be & 0xffu) << 24u) | (((be >> 8u) & 0xffu) << 16u)
         | (((be >> 16u) & 0xffu) << 8u) | ((be >> 24u) & 0xffu);
    // Wait: SHA state words are BE meaning digest bytes are:
    // byte0 = sw[0]>>24, byte1=sw[0]>>16, ...
    // RIPEMD LE word0 = byte0 | byte1<<8 | byte2<<16 | byte3<<24
    w[i] = ((be >> 24u) & 0xffu)
         | (((be >> 16u) & 0xffu) << 8u)
         | (((be >> 8u) & 0xffu) << 16u)
         | ((be & 0xffu) << 24u);
  }
  w[8] = 0x00000080u;
  for (var i: u32 = 9u; i < 14u; i++) { w[i] = 0u; }
  w[14] = 256u; // bit length = 32*8 = 256
  w[15] = 0u;

  var h0=0x67452301u; var h1=0xEFCDAB89u; var h2=0x98BADCFEu;
  var h3=0x10325476u; var h4=0xC3D2E1F0u;
  var al=h0; var bl=h1; var cl=h2; var dl=h3; var el=h4;
  var ar=h0; var br=h1; var cr=h2; var dr=h3; var er=h4;
  for (var j: u32 = 0u; j < 80u; j++) {
    var t = al + ripemf(j, bl, cl, dl) + w[ZL[j]] + KL[j >> 4u];
    t = rotl(t, SL[j]) + el;
    al = el; el = dl; dl = rotl(cl, 10u); cl = bl; bl = t;
    t = ar + ripemf(79u - j, br, cr, dr) + w[ZR[j]] + KR[j >> 4u];
    t = rotl(t, SR[j]) + er;
    ar = er; er = dr; dr = rotl(cr, 10u); cr = br; br = t;
  }
  let t = h1 + cl + dr;
  h1 = h2 + dl + er;
  h2 = h3 + el + ar;
  h3 = h4 + al + br;
  h4 = h0 + bl + cr;
  h0 = t;
  return array<u32, 5>(h0, h1, h2, h3, h4);
}

// Convert RIPEMD LE words to 5 BE words matching hash160ToWordsBE
fn h160_to_be_words(h: array<u32, 5>) -> array<u32, 5> {
  var out: array<u32, 5>;
  for (var i: u32 = 0u; i < 5u; i++) {
    let le = h[i];
    // bytes of hash160 are LE word bytes in order
    let b0 = le & 0xffu;
    let b1 = (le >> 8u) & 0xffu;
    let b2 = (le >> 16u) & 0xffu;
    let b3 = (le >> 24u) & 0xffu;
    out[i] = (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3;
  }
  return out;
}

fn hash160_match(prefix: u32, x: Fe) -> bool {
  let sha = sha256_pubkey33(prefix, x);
  let hle = ripemd160_sha_words(sha);
  let hbe = h160_to_be_words(hle);
  return hbe[0]==U.target0 && hbe[1]==U.target1 && hbe[2]==U.target2
      && hbe[3]==U.target3 && hbe[4]==U.target4;
}

fn odd_y(y: Fe) -> bool {
  return (y[0] & 1u) == 1u;
}

fn report_found(key_offset: u32) {
  // key = start_key + key_offset
  let old = atomicCompareExchangeWeak(&result[0], 0u, 1u);
  if (old.exchanged) {
    // compute key limbs = sk + offset
    var k: Fe = Fe(U.sk0, U.sk1, U.sk2, U.sk3, U.sk4, U.sk5, U.sk6, U.sk7);
    var off = key_offset;
    var carry: u32 = 0u;
    // add off (u32) into k
    let s0 = k[0] + off;
    carry = select(0u, 1u, s0 < k[0]);
    k[0] = s0;
    for (var i: u32 = 1u; i < 8u; i++) {
      let si = k[i] + carry;
      carry = select(0u, 1u, si < k[i]);
      k[i] = si;
    }
    for (var i: u32 = 0u; i < 8u; i++) {
      atomicStore(&result[1u + i], k[i]);
    }
  }
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= U.thread_count) { return; }
  if (atomicLoad(&result[0]) == 1u) { return; }

  let kpt = U.keys_per_thread;
  let G = load_G();
  var P = jac_from_aff(load_base());

  var idx = tid;
  for (var bit: u32 = 0u; bit < 16u; bit++) {
    if ((idx & 1u) != 0u) {
      let aff = load_offset_aff(bit);
      P = jac_add_aff(P, aff);
    }
    idx = idx >> 1u;
  }

  var produced: u32 = 0u;
  loop {
    if (produced >= kpt) { break; }
    if (atomicLoad(&result[0]) == 1u) { return; }

    let nbatch = min(BATCH, kpt - produced);
    var zs: array<Fe, 48>;
    var xs: array<Fe, 48>;
    var ys: array<Fe, 48>;
    var valid: array<u32, 48>;
    var cur = P;
    for (var i: u32 = 0u; i < nbatch; i++) {
      xs[i] = cur.x; ys[i] = cur.y; zs[i] = cur.z;
      valid[i] = select(0u, 1u, !fe_is_zero(cur.z));
      if (fe_is_zero(zs[i])) { zs[i] = FE_ONE; }
      cur = jac_add_aff(cur, G);
    }
    P = cur;

    var invs: array<Fe, 48>;
    batch_invert(&zs, &invs, nbatch);

    for (var i: u32 = 0u; i < nbatch; i++) {
      if (valid[i] == 0u) { continue; }
      let zinv = invs[i];
      let z2 = fe_sqr(zinv);
      let z3 = fe_mul(z2, zinv);
      let ax = fe_mul(xs[i], z2);
      let ay = fe_mul(ys[i], z3);
      let prefix = select(0x02u, 0x03u, odd_y(ay));
      if (hash160_match(prefix, ax)) {
        let key_offset = tid * kpt + produced + i;
        report_found(key_offset);
        return;
      }
    }
    produced += nbatch;
  }
}
