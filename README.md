# 本地 WebGPU Bitcoin Puzzle 搜尋器

純瀏覽器、純本地嘅 Bitcoin Puzzle（例如 #71）暴力搜尋工具。用你部機嘅 GPU（WebGPU）行 secp256k1 連續點加 + HASH160 比對。

## 零抽成承諾

- 中獎私鑰**只會**顯示喺你嘅瀏覽器（hex + compressed WIF）
- **唔會**自動分任何比例去開發者地址
- **唔會**將私鑰上傳／加密交去伺服器
- **唔會**自動廣播交易

如果你之後自己寫掃款交易，應將 **100%（扣礦工費）** 轉去你自己嘅地址。

## 瀏覽器要求

- **Chrome** 或 **Edge**（需要 WebGPU）
- Firefox 對 WebGPU 支援可能唔完整
- 建議桌面 GPU；集成顯卡都得，只係慢啲

### Linux 提示（Vulkan）

如果 Chrome 偵測唔到 WebGPU，可以試：

```bash
google-chrome --enable-features=Vulkan --enable-unsafe-webgpu
```

或者喺 `chrome://flags` 打開相關 WebGPU / Vulkan 旗標。

## 點樣開

喺專案目錄開一個靜態伺服器（唔好直接用 `file://`，ES modules／WGSL fetch 可能被擋）：

```bash
cd btc-puzzle-webgpu
python3 -m http.server 8765
```

然後用 Chrome／Edge 打開：

[http://127.0.0.1:8765/](http://127.0.0.1:8765/)

## 使用流程

1. 等自測全部通過（Start 先會解鎖）
2. 選謎題（預設 Puzzle #71），或 DEMO／自訂
3. （可選）填你嘅收款地址——只作提示，唔會被抽
4. 撳「開始搜尋」
5. 若命中：畫面會顯示私鑰；請小心用私人中繼歸集

可用「CPU DEMO 命中演練」喺冇 GPU 時都睇到命中 UI。

## Puzzle #71 參數（預設）

| 項目 | 值 |
|------|-----|
| 地址 | `1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU` |
| HASH160 | `f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8` |
| 範圍 | `0x400000000000000000` … `0x7fffffffffffffffff`（2^70 … 2^71−1） |

## 誠實勝算

Puzzle 71 有 **2^70 ≈ 1.18×10²¹** 個候選 key。就算瀏覽器 GPU 去到數千萬 keys/s，中位數時間仍然以「天文數字年」計。呢個係公開彩票式謎題，**唔係**投資工具。

## 架構（簡述）

1. CPU（`@noble/secp256k1`）計每個 chunk 起點 `P = k·G`
2. GPU 用 Jacobian 連續 `+G` 行走（唔係每把 key 做完整 scalar mul）
3. 每批 ~16 個 Z 做 Montgomery batch inversion
4. 壓縮公鑰 → SHA-256 → RIPEMD-160 → 同目標 HASH160 比較
5. Dispatch 大小自動校準到約 250ms，並存 `localStorage`

## 額外自測頁

`gpu-selftest.html`：獨立 GPU 向量測試頁（開發用）。

## 目錄

```
btc-puzzle-webgpu/
  index.html          # UI（繁中 HK）
  app.js              # WebGPU 調度 / 自測 / 搜尋
  shaders/search.wgsl # secp256k1 + SHA256 + RIPEMD160
  lib/noble-secp256k1.js
  lib/crypto-utils.js
  lib/ripemd160.js
  README.md
```

## 授權

- 本專案碼可自由使用
- `lib/noble-secp256k1.js` 跟 Paul Miller 的 MIT License（見 `lib/noble-LICENSE.txt`）
