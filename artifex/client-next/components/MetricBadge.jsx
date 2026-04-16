"use client";

export default function MetricBadge({ value, label, decimals = 4 }) {
  if (value === null || value === undefined) {
    return null;
  }

  const displayValue =
    typeof value === "number" ? value.toFixed(decimals) : value;

  return (
    <div className="metric-badge flex flex-col items-center rounded-lg px-3 py-2 min-w-[80px]">
      <span className="text-sm font-semibold text-[var(--color-gold-light)]">
        {displayValue}
      </span>

      <span className="text-[10px] text-gray-400 text-center mt-0.5">
        {label}
      </span>
    </div>
  );
}