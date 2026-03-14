"use client";

import { useState } from "react";

/**
 * BeforeAfterSlider — drag to compare damaged vs. restored.
 */
export default function BeforeAfterSlider({ originalSrc, restoredSrc }) {
  const [pos, setPos] = useState(50);

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x =
      (e.clientX || (e.touches && e.touches[0].clientX) || 0) - rect.left;
    setPos(Math.max(0, Math.min(100, (x / rect.width) * 100)));
  };

  return (
    <div
      onMouseMove={handleMove}
      onTouchMove={handleMove}
      className="relative overflow-hidden rounded-lg cursor-ew-resize select-none bg-black"
      style={{ aspectRatio: "1", width: "100%" }}
    >
      {/* Restored (full background) */}
      <img
        src={restoredSrc}
        alt="Restored"
        className="absolute inset-0 w-full h-full object-cover"
      />
      {/* Damaged (clipped to left of slider) */}
      <div
        className="absolute top-0 left-0 h-full overflow-hidden"
        style={{ width: `${pos}%` }}
      >
        <img
          src={originalSrc}
          alt="Damaged"
          className="absolute top-0 left-0 h-full object-cover"
          style={{ width: `${10000 / Math.max(pos, 1)}%`, maxWidth: "none" }}
        />
      </div>
      {/* Slider line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-[var(--color-gold)]"
        style={{ left: `${pos}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-[var(--color-gold)] rounded-full w-7 h-7 flex items-center justify-center text-xs text-black font-bold">
          ⟷
        </div>
      </div>
      {/* Labels */}
      <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-white">
        DAMAGED
      </div>
      <div className="absolute bottom-2 right-2 bg-[var(--color-gold)]/90 px-2 py-0.5 rounded text-[10px] text-black font-semibold">
        RESTORED
      </div>
    </div>
  );
}
