"use client";

import { useState } from "react";
import { MoveHorizontal } from "lucide-react";

export default function BeforeAfterSlider({ originalSrc, restoredSrc }) {
  const [position, setPosition] = useState(50);

  const handleMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const clientX = event.clientX || event.touches?.[0]?.clientX || 0;
    const x = clientX - rect.left;

    const nextPosition = (x / rect.width) * 100;
    setPosition(Math.max(0, Math.min(100, nextPosition)));
  };

  return (
    <div
      onMouseMove={handleMove}
      onTouchMove={handleMove}
      className="relative overflow-hidden rounded-lg cursor-ew-resize select-none bg-black"
      style={{ aspectRatio: "1", width: "100%" }}
    >
      <img
        src={restoredSrc}
        alt="Restored artwork"
        className="absolute inset-0 w-full h-full object-cover"
      />

      <div
        className="absolute top-0 left-0 h-full overflow-hidden"
        style={{ width: `${position}%` }}
      >
        <img
          src={originalSrc}
          alt="Damaged artwork"
          className="absolute top-0 left-0 h-full object-cover"
          style={{
            width: `${10000 / Math.max(position, 1)}%`,
            maxWidth: "none",
          }}
        />
      </div>

      <div
        className="absolute top-0 bottom-0 w-0.5 bg-[var(--color-gold)]"
        style={{ left: `${position}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-[var(--color-gold)] rounded-full w-7 h-7 flex items-center justify-center text-black">
          <MoveHorizontal size={16} aria-hidden="true" />
        </div>
      </div>

      <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-white">
        DAMAGED
      </div>

      <div className="absolute bottom-2 right-2 bg-[var(--color-gold)]/90 px-2 py-0.5 rounded text-[10px] text-black font-semibold">
        RESTORED
      </div>
    </div>
  );
}