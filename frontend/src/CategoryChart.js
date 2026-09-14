import React from 'react';
import { ResponsiveContainer, Tooltip, Treemap } from 'recharts';

// Same tokens as App.css — kept in sync manually since SVG fills can't read CSS custom properties.
const COLORS = {
  light: { accent: '#2a78d6', sensitive: '#d03b3b', text: '#52514e', surface: '#fcfcfb' },
  dark: { accent: '#3987e5', sensitive: '#e66767', text: '#c3c2b7', surface: '#1a1a19' },
};

function Tile({ x, y, width, height, name, sensitive, depth, value, root, colors }) {
  if (width <= 0 || height <= 0) return null;

  const percent = root && root.value ? ((value || 0) / root.value) * 100 : 0;
  const label = `${name} — ${percent.toFixed(1)}%`;

  if (depth === 1) {
    const showLabel = width > 55 && height > 20;
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={sensitive ? colors.sensitive : colors.accent}
          stroke={colors.surface}
          strokeWidth={2}
          rx={3}
        />
        {showLabel && (
          <text x={x + 8} y={y + 18} fill="#fff" fontSize={12} fontWeight={700}>
            {name}
          </text>
        )}
        {showLabel && height > 36 && (
          <text x={x + 8} y={y + 34} fill="#fff" fontSize={11} fillOpacity={0.85}>
            {percent.toFixed(1)}%
          </text>
        )}
      </g>
    );
  }

  // Sub-category tile, nested inside its parent.
  const showLabel = width > 45 && height > 16;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill="none"
        stroke={colors.surface}
        strokeWidth={1.5}
      />
      {showLabel && (
        <text x={x + 5} y={y + 13} fill="#fff" fontSize={9.5} fillOpacity={0.95}>
          {name}
        </text>
      )}
      {showLabel && height > 30 && (
        <text x={x + 5} y={y + 26} fill="#fff" fontSize={9} fillOpacity={0.8}>
          {percent.toFixed(1)}%
        </text>
      )}
      <title>{label}</title>
    </g>
  );
}

function CustomTooltip({ active, payload, colors }) {
  if (!active || !payload || !payload.length) return null;
  const node = payload[0].payload;
  if (!node || node.children) return null; // only show tooltip for leaf sub-category tiles

  return (
    <div
      style={{
        background: colors.surface,
        border: `1px solid ${colors.text}`,
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: '0.82rem',
        color: colors.text,
      }}
    >
      <div style={{ fontWeight: 600 }}>{node.name}</div>
      <div>{node.size} matched record{node.size === 1 ? '' : 's'}</div>
    </div>
  );
}

export default function CategoryChart({ data, theme }) {
  if (!data || data.length === 0) return null;

  const colors = theme === 'dark' ? COLORS.dark : COLORS.light;
  const hasSensitive = data.some((d) => d.sensitive);

  return (
    <div>
      <div style={{ width: '100%', height: 340 }}>
        <ResponsiveContainer>
          <Treemap
            data={data}
            dataKey="size"
            aspectRatio={4 / 3}
            stroke={colors.surface}
            isAnimationActive
            animationDuration={700}
            animationEasing="ease-out"
            content={<Tile colors={colors} />}
          >
            <Tooltip content={<CustomTooltip colors={colors} />} />
          </Treemap>
        </ResponsiveContainer>
      </div>
      {hasSensitive && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: colors.text, marginTop: 4 }}>
          <span
            style={{
              display: 'inline-block',
              width: 10,
              height: 10,
              borderRadius: 3,
              background: colors.sensitive,
            }}
          />
          Sensitive category
        </div>
      )}
    </div>
  );
}
