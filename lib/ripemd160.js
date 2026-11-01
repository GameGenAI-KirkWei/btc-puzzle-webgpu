/** RIPEMD-160 (public domain algorithm). ESM. */
const KL = new Uint32Array([0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E]);
const KR = new Uint32Array([0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000]);
const ZL = new Uint8Array([
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
  7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
  3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
  1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
  4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]);
const ZR = new Uint8Array([
  5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
  6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
  15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
  8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
  12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]);
const SL = new Uint8Array([
  11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
  7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
  11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
  11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
  9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]);
const SR = new Uint8Array([
  8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
  9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
  9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
  15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
  8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]);

function rotl(x,n){return (x<<n)|(x>>> (32-n));}
function f(j,x,y,z){
  if(j<16) return x^y^z;
  if(j<32) return (x&y)|(~x&z);
  if(j<48) return (x|~y)^z;
  if(j<64) return (x&z)|(y&~z);
  return x^(y|~z);
}

export function ripemd160(data){
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const bitLen = BigInt(bytes.length) * 8n;
  const withOne = bytes.length + 1;
  const padLen = (withOne % 64 <= 56) ? (56 - (withOne % 64)) : (120 - (withOne % 64));
  const buf = new Uint8Array(withOne + padLen + 8);
  buf.set(bytes);
  buf[bytes.length] = 0x80;
  const dv = new DataView(buf.buffer);
  dv.setUint32(buf.length-8, Number(bitLen & 0xffffffffn), true);
  dv.setUint32(buf.length-4, Number((bitLen >> 32n) & 0xffffffffn), true);

  let h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
  const w = new Uint32Array(16);
  for(let i=0;i<buf.length;i+=64){
    for(let j=0;j<16;j++) w[j]=dv.getUint32(i+j*4,true);
    let al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    let ar=h0,br=h1,cr=h2,dr=h3,er=h4;
    for(let j=0;j<80;j++){
      let t = (al + f(j,bl,cl,dl) + w[ZL[j]] + KL[j>>4]) >>> 0;
      t = (rotl(t, SL[j]) + el) >>> 0;
      al=el; el=dl; dl=rotl(cl,10); cl=bl; bl=t;
      t = (ar + f(79-j,br,cr,dr) + w[ZR[j]] + KR[j>>4]) >>> 0;
      t = (rotl(t, SR[j]) + er) >>> 0;
      ar=er; er=dr; dr=rotl(cr,10); cr=br; br=t;
    }
    const t = (h1 + cl + dr) >>> 0;
    h1 = (h2 + dl + er) >>> 0;
    h2 = (h3 + el + ar) >>> 0;
    h3 = (h4 + al + br) >>> 0;
    h4 = (h0 + bl + cr) >>> 0;
    h0 = t;
  }
  const out = new Uint8Array(20);
  const o = new DataView(out.buffer);
  o.setUint32(0,h0,true); o.setUint32(4,h1,true); o.setUint32(8,h2,true);
  o.setUint32(12,h3,true); o.setUint32(16,h4,true);
  return out;
}
