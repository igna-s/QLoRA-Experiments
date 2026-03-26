import { useState } from "react";

const FROZEN_COLOR = "#1e3a5f";
const LORA_COLOR = "#ff6b2b";
const ACTIVE_COLOR = "#00e5ff";

export default function App() {
  const [selected, setSelected] = useState(null);
  const [hoveredLayer, setHoveredLayer] = useState(null);

  const layers = [1, 2, 3, "...", 14, "...", 27, 28];
  const modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

  const info = {
    base: {
      title: "Modelo Base — Qwen2.5-7B",
      color: "#00e5ff",
      content: [
        "• 7,600 millones de parámetros",
        "• 28 capas Transformer",
        "• Congelado en 4-bit NF4 (QLoRA)",
        "• NO se actualiza durante el entrenamiento",
        "• Ocupa ~4 GB de VRAM gracias a la cuantización",
      ],
    },
    lora: {
      title: "Adaptadores LoRA (añadidos)",
      color: "#ff6b2b",
      content: [
        "• ~80 millones de parámetros NUEVOS",
        "• Solo 1.1% del total — muy eficiente",
        "• 7 módulos × 28 capas = 196 adaptadores",
        "• Cada uno: matriz A (r×d_in) + matriz B (d_out×r)",
        "• r=32, alpha=64",
        "• En fp16, SÍ se actualizan con gradientes",
      ],
    },
    math: {
      title: "¿Cómo funciona?",
      color: "#a78bfa",
      content: [
        "Para cada proyección (ej. q_proj):",
        "",
        "  salida = W_original · x   ← CONGELADO",
        "         + B · A · x        ← LoRA ENTRENADO",
        "",
        "Al inicio: B·A = 0 (no rompe el modelo)",
        "Al final:  B·A = el 'delta' aprendido",
        "Se guarda solo B y A, no el modelo entero",
      ],
    },
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#07080f",
      color: "#e0e6f0",
      fontFamily: "'Courier New', monospace",
      padding: "32px 24px",
      overflowX: "hidden",
    }}>
      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <div style={{ fontSize: 11, letterSpacing: 6, color: "#4a5568", marginBottom: 8 }}>
          ARQUITECTURA
        </div>
        <h1 style={{
          fontSize: "clamp(22px, 4vw, 36px)",
          fontWeight: 900,
          margin: 0,
          background: "linear-gradient(135deg, #00e5ff, #a78bfa)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: 2,
        }}>
          Qwen2.5-7B + QLoRA
        </h1>
        <div style={{ color: "#4a5568", fontSize: 12, marginTop: 8 }}>
          Hacé click en cualquier elemento para ver detalles
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 24, justifyContent: "center", marginBottom: 36, flexWrap: "wrap" }}>
        {[
          { color: FROZEN_COLOR, border: "#00e5ff", label: "Congelado (4-bit NF4)", key: "base" },
          { color: "#2d1a0e", border: LORA_COLOR, label: "LoRA añadido (fp16, entrenado)", key: "lora" },
        ].map(({ color, border, label, key }) => (
          <div
            key={key}
            onClick={() => setSelected(selected === key ? null : key)}
            style={{
              display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
              padding: "6px 14px", borderRadius: 6,
              border: `1px solid ${selected === key ? border : "#222"}`,
              background: selected === key ? `${border}11` : "transparent",
              transition: "all 0.2s",
            }}
          >
            <div style={{ width: 14, height: 14, borderRadius: 3, background: color, border: `2px solid ${border}` }} />
            <span style={{ fontSize: 12, color: "#9aa5b4" }}>{label}</span>
          </div>
        ))}
        <div
          onClick={() => setSelected(selected === "math" ? null : "math")}
          style={{
            display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
            padding: "6px 14px", borderRadius: 6,
            border: `1px solid ${selected === "math" ? "#a78bfa" : "#222"}`,
            background: selected === "math" ? "#a78bfa11" : "transparent",
            transition: "all 0.2s",
          }}
        >
          <span style={{ fontSize: 14 }}>∑</span>
          <span style={{ fontSize: 12, color: "#9aa5b4" }}>Fórmula matemática</span>
        </div>
      </div>

      {/* Info Panel */}
      {selected && (
        <div style={{
          margin: "0 auto 32px",
          maxWidth: 600,
          background: `${info[selected].color}0f`,
          border: `1px solid ${info[selected].color}44`,
          borderRadius: 10,
          padding: "18px 24px",
          animation: "fadeIn 0.2s ease",
        }}>
          <div style={{ color: info[selected].color, fontWeight: 700, fontSize: 14, marginBottom: 12 }}>
            {info[selected].title}
          </div>
          {info[selected].content.map((line, i) => (
            <div key={i} style={{
              fontSize: 12, color: "#c5d0e0", lineHeight: 1.8,
              fontFamily: line.startsWith(" ") ? "'Courier New', monospace" : "inherit",
              whiteSpace: "pre",
            }}>
              {line}
            </div>
          ))}
        </div>
      )}

      {/* Main diagram */}
      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* Input */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
          <div style={{
            background: "#111827", border: "1px solid #374151",
            borderRadius: 8, padding: "8px 32px", fontSize: 12, color: "#6b7280",
          }}>
            Token Input (texto)
          </div>
        </div>
        <Arrow />

        {/* Embedding */}
        <CenteredBlock
          label="Embedding Layer"
          sublabel="vocab_size → 3584"
          color={FROZEN_COLOR}
          border="#00e5ff"
        />
        <Arrow />

        {/* Layers */}
        <div style={{ border: "1px dashed #1e3a5f", borderRadius: 12, padding: "16px 8px", marginBottom: 8 }}>
          <div style={{ textAlign: "center", fontSize: 10, color: "#4a5568", marginBottom: 12, letterSpacing: 3 }}>
            × 28 CAPAS TRANSFORMER
          </div>

          {/* Layer rows */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {layers.map((layer, idx) => (
              layer === "..." ? (
                <div key={idx} style={{ textAlign: "center", color: "#374151", fontSize: 18, letterSpacing: 8 }}>
                  ·  ·  ·
                </div>
              ) : (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredLayer(layer)}
                  onMouseLeave={() => setHoveredLayer(null)}
                  style={{
                    background: hoveredLayer === layer ? "#0d1b2a" : "#0a0f1a",
                    border: `1px solid ${hoveredLayer === layer ? "#1e3a5f" : "#111827"}`,
                    borderRadius: 8, padding: "10px 12px",
                    transition: "all 0.15s", cursor: "default",
                  }}
                >
                  {/* Layer label */}
                  <div style={{ fontSize: 10, color: "#4a5568", marginBottom: 8, letterSpacing: 2 }}>
                    CAPA {layer}
                  </div>

                  {/* Modules grid */}
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {modules.map((mod) => {
                      const isAttn = ["q_proj","k_proj","v_proj","o_proj"].includes(mod);
                      return (
                        <div key={mod} style={{ flex: "1 1 90px", minWidth: 80 }}>
                          {/* Frozen base */}
                          <div style={{
                            background: FROZEN_COLOR,
                            border: "1px solid #00e5ff44",
                            borderRadius: "4px 4px 0 0",
                            padding: "4px 6px",
                            fontSize: 10,
                            color: "#60a5fa",
                            textAlign: "center",
                          }}>
                            {mod}
                            <div style={{ fontSize: 8, color: "#1e4080", marginTop: 2 }}>
                              {isAttn ? "W_attn" : "W_ffn"} ❄️
                            </div>
                          </div>
                          {/* LoRA adapters */}
                          <div style={{
                            background: "#2d1a0e",
                            border: "1px solid #ff6b2b88",
                            borderTop: "none",
                            borderRadius: "0 0 4px 4px",
                            padding: "3px 6px",
                            fontSize: 9,
                            color: LORA_COLOR,
                            textAlign: "center",
                          }}>
                            A [{32}×d] + B [d×{32}]
                            <div style={{ fontSize: 8, color: "#cc4400", marginTop: 1 }}>
                              LoRA r=32 ✦ entrenado
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* RMSNorm + output */}
                  <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                    <div style={{
                      flex: 1, background: FROZEN_COLOR, border: "1px solid #00e5ff22",
                      borderRadius: 4, padding: "4px 8px", fontSize: 10, color: "#4a7fa5", textAlign: "center",
                    }}>
                      RMSNorm ❄️
                    </div>
                    <div style={{
                      flex: 1, background: FROZEN_COLOR, border: "1px solid #00e5ff22",
                      borderRadius: 4, padding: "4px 8px", fontSize: 10, color: "#4a7fa5", textAlign: "center",
                    }}>
                      Residual ❄️
                    </div>
                  </div>
                </div>
              )
            ))}
          </div>
        </div>

        <Arrow />
        {/* LM Head */}
        <CenteredBlock
          label="LM Head (output)"
          sublabel="3584 → vocab_size"
          color={FROZEN_COLOR}
          border="#00e5ff"
        />
        <Arrow />

        {/* Output */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 40 }}>
          <div style={{
            background: "#111827", border: "1px solid #374151",
            borderRadius: 8, padding: "8px 32px", fontSize: 12, color: "#6b7280",
          }}>
            Probabilidades → Token generado
          </div>
        </div>

        {/* Stats bar */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: 12, marginTop: 8,
        }}>
          {[
            { label: "Parámetros totales", value: "7,600 M", color: "#00e5ff" },
            { label: "Parámetros LoRA", value: "~80 M", color: LORA_COLOR },
            { label: "% entrenado", value: "~1.1%", color: "#a78bfa" },
            { label: "Módulos por capa", value: "7 módulos", color: "#34d399" },
            { label: "Capas × módulos", value: "28 × 7 = 196", color: "#fbbf24" },
            { label: "Cuantización base", value: "4-bit NF4", color: "#60a5fa" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              background: "#0a0f1a", border: `1px solid ${color}33`,
              borderRadius: 8, padding: "12px 16px", textAlign: "center",
            }}>
              <div style={{ fontSize: 18, fontWeight: 900, color }}>{value}</div>
              <div style={{ fontSize: 10, color: "#4a5568", marginTop: 4 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

function Arrow() {
  return (
    <div style={{ display: "flex", justifyContent: "center", margin: "4px 0", color: "#1e3a5f", fontSize: 18 }}>
      ↓
    </div>
  );
}

function CenteredBlock({ label, sublabel, color, border }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
      <div style={{
        background: color, border: `1px solid ${border}55`,
        borderRadius: 8, padding: "10px 40px", textAlign: "center",
        minWidth: 200,
      }}>
        <div style={{ fontSize: 13, color: "#90cdf4", fontWeight: 700 }}>{label}</div>
        <div style={{ fontSize: 10, color: "#4a7fa5", marginTop: 3 }}>{sublabel} ❄️</div>
      </div>
    </div>
  );
}
